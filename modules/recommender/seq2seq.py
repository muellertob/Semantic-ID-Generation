import torch
import torch.nn as nn
import logging
from transformers import T5Config, T5Model

logger = logging.getLogger(__name__)

class TigerSeq2Seq(nn.Module):
    def __init__(self, codebook_layers=3, codebook_size=256, user_tokens=2000, d_model=512, d_kv=64, d_ff=2048, num_layers=4, num_heads=6, dropout=0.1, activation_fn="relu"):
        super().__init__()
        self.codebook_layers = codebook_layers
        self.codebook_size = codebook_size

        # VOCABULARY SIZE
        self.semantic_vocab_size = codebook_layers * codebook_size
        self.collision_vocab_size = codebook_size # additional tokens for ID collisions
        self.user_vocab_size = user_tokens

        # only semantic, collision, and special tokens can be output by the decoder
        self.output_vocab_size = (self.semantic_vocab_size + 
                                  self.collision_vocab_size + 
                                  3)
        
        # total vocab includes user tokens for the shared embedding space
        self.total_vocab = self.output_vocab_size + self.user_vocab_size
        
        # set special token indices (placed directly after collision tokens)
        self.pad_idx = self.output_vocab_size - 3
        self.bos_idx = self.output_vocab_size - 2
        self.eos_idx = self.output_vocab_size - 1

        # OFFSETS: shift tokens to avoid overlaps in the shared embedding space
        semantic_offsets = torch.arange(0, codebook_layers * codebook_size, step=codebook_size)
        collision_start = torch.tensor([self.semantic_vocab_size])

        # register as buffer to move with the model to device
        self.register_buffer('item_offsets', 
                             torch.cat([semantic_offsets, collision_start]))
        
        # collision tokens start after semantic tokens
        self.collision_offset = self.semantic_vocab_size

        # user tokens start after the output vocab
        self.user_offset = self.output_vocab_size

        # LOSS FUNCTION
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_idx, reduction='none')

        # TRANSFORMER MODEL
        # using T5 as the backbone seq2seq model like in the original TIGER paper
        config = T5Config(
            vocab_size=self.total_vocab,
            d_model=d_model,                    # TIGER: 128 (Input Dimension)
            d_kv=d_kv,                          # TIGER: 64
            d_ff=d_ff,                          # TIGER: 1024 (MLP Dimension)
            num_layers=num_layers,              # TIGER: 4 layers for both encoder and decoder
            num_decoder_layers=num_layers,
            num_heads=num_heads,                # TIGER: 6 self-attention heads
            dropout_rate=dropout,               # TIGER: 0.1 dropout
            feed_forward_proj=activation_fn,    # TIGER: relu activation
            pad_token_id=self.pad_idx,
            decoder_start_token_id=self.bos_idx,
            eos_token_id=self.eos_idx,
            use_cache=True
        )

        self.backbone = T5Model(config)

        # OUTPUT PROJECTION: maps hidden state back to output vocab probabilities
        self.output_projection = nn.Linear(d_model, self.output_vocab_size, bias=False)
        
        # codebook buffer for constrained generation
        # will be set via set_codebooks()
        self.register_buffer('codebooks', None)
        
        # trie buffers
        self.register_buffer('trie_transitions', None) # [Nodes, Vocab] -> Next Node
        self.register_buffer('trie_masks', None)       # [Nodes, Vocab] -> Allowed Logit Mask

    def build_trie(self, codebooks):
        """
        Builds a prefix tree (Trie) from the valid codebooks and registers 
        vectorized transition and mask tables for fast lookups.
        
        :param codebooks: Tensor of shape [Num_Items, Codebook_Layers + 1] containing shifted token IDs.
        """
        logger.info(f"Building Trie Index for {len(codebooks)} items...")
        
        # initialize Trie with root node
        trie = [{'children': {}, 'parent': -1, 'value': -1}]
        codebook_list = codebooks.cpu().tolist()
        
        # iterate through each codebook sequence and insert into the trie
        for seq in codebook_list:
            # start at the root for each sequence
            current_node = 0
            for token in seq:
                # check if the token already exists as a child of the current node
                if token not in trie[current_node]['children']:
                    new_node_idx = len(trie)
                    # append new node to the trie with parent and value information
                    trie.append({'children': {}, 'parent': current_node, 'value': token})
                    # link the current node to the new node via the token
                    trie[current_node]['children'][token] = new_node_idx
                # move to the child node for the next token
                current_node = trie[current_node]['children'][token]
                
        num_nodes = len(trie)
        # allocate one extra node for the "sink" state to handle invalid transitions safely
        sink_node = num_nodes
        total_nodes = num_nodes + 1
        
        logger.info(f"Trie built with {num_nodes} valid nodes and 1 sink node.")
        
        # convert to dense tensor tables
        # initialize with sink_node (invalid state traps here)
        # NOTE: replace with sparse representation if memory becomes an issue with large codebooks
        transition_tensor = torch.full((total_nodes, self.output_vocab_size), sink_node, dtype=torch.int32)
        # initialize mask with False (invalid token)
        mask_tensor = torch.zeros((total_nodes, self.output_vocab_size), dtype=torch.bool)
        
        # populate the matrices
        for i, node in enumerate(trie):
            for token, child_idx in node['children'].items():
                if token < self.output_vocab_size:
                    transition_tensor[i, token] = child_idx
                    mask_tensor[i, token] = True
                    
        # register buffers (will be saved alongside parameters and moved to device with the model)
        device = self.item_offsets.device
        self.register_buffer('trie_transitions', transition_tensor.to(device=device))
        self.register_buffer('trie_masks', mask_tensor.to(device=device))

    def set_codebooks(self, codebooks):
        """
        Registers the valid Semantic IDs for constrained generation.
        Applies offsets to match the model's vocabulary space.
        
        :param codebooks: Tensor of shape [Num_Items, Codebook_Layers + 1] containing raw indices.
        """
        device = self.item_offsets.device
        if codebooks.device != device:
            codebooks = codebooks.to(device=device)
            
        # ensure dimensionality matches for broadcasting
        # codebooks: [N, L]; item_offsets: [L]
        if codebooks.shape[1] != self.item_offsets.shape[0]:
             raise ValueError(f"Codebook shape {codebooks.shape} does not match item offsets shape {self.item_offsets.shape}")
        
        # apply offsets to raw codebook indices to match vocabulary   
        shifted_codebooks = codebooks + self.item_offsets.unsqueeze(0)
        self.register_buffer('codebooks', shifted_codebooks)
        
        # build the Trie index for fast constrained generation
        self.build_trie(shifted_codebooks)

    def process_input_tuples(self, input_tuples, user_ids=None):
        """
        Processes input tuples (history or target) by applying offsets, handling padding, 
        and optionally prepending user tokens to create a flat sequence of input IDs.

        This function serves two main purposes:
        1. Encoder Input: Processes user history tuples and prepends the user ID token.
        2. Decoder Target: Processes target item tuples (without user IDs) for training.

        :param input_tuples: Tensor of shape [Batch, Seq_Len, Tuple_Size] containing discrete codes.
                             Padding should be represented by -1.
        :param user_ids: Optional Tensor of shape [Batch] containing user IDs. 
                         If provided, a user token is prepended to the sequence.
        :return: Flattened Tensor of shape [Batch, Sequence_Length] (plus 1 if user_ids provided)
                 containing global vocabulary indices.
        """
        batch_size, seq_len, _ = input_tuples.shape

        # IDENTIFY PADDING
        # collate function is expected to pad with -1
        is_padding = (input_tuples == -1)
        # replace padding with zeros for offset addition
        clean_tuples = input_tuples.masked_fill(is_padding, 0)

        # PROCESS TOKENS
        # offset is broadcasted to match input_tuples shape
        offset_tuples = clean_tuples + self.item_offsets.view(1, 1, -1)

        # RESTORE PADDING
        offset_tuples = offset_tuples.masked_fill(is_padding, self.pad_idx)

        flat_sequence = offset_tuples.view(batch_size, -1) # [B, T, 4] -> [B, T * 4]

        if user_ids is None:
            # shortcut when no user IDs are provided
            return flat_sequence
        
        # PROCESS USER TOKENS
        hashed_user = user_ids % self.user_vocab_size # hashing trick maps to user token range
        user_tokens = (hashed_user + self.user_offset).unsqueeze(1) # [B, 1]

        # CONCATENATE USER AND TUPLES
        full_sequence = torch.cat([user_tokens, flat_sequence], dim=1) # [B, 1 + T * 4]

        return full_sequence

    def forward(self, history_tuples, target_tuples=None, user_ids=None):
        """
        Encodes user history and either computes training loss or returns encoder states.

        :param history_tuples: A tensor of Semantic ID tuples representing the user's history.
        :param target_tuples: An optional tensor of Semantic ID tuples serving as prediction targets.
        :param user_ids: An optional tensor of user IDs.
        """
        device = history_tuples.device
        batch_size = history_tuples.size(0)

        # PREPARE ENCODER INPUT
        # flatten history and add offsets
        encoder_input = self.process_input_tuples(history_tuples, user_ids)
        # fill attention mask for padding tokens (1 for real tokens, 0 for padding)
        encoder_attention_mask = (encoder_input != self.pad_idx).long()

        # PREPARE DECODER INPUT
        decoder_input = None
        decoder_attention_mask = None
        loss = None

        if target_tuples is not None:
            # prepare target sequence
            raw_target_ids = self.process_input_tuples(target_tuples, user_ids=None)

            # TEACHER FORCING: decoder input is target sequence shifted right
            bos_tokens = torch.full(
                (batch_size, 1), 
                self.bos_idx,
                dtype=torch.long,
                device=device
            )

            decoder_input = torch.cat([bos_tokens, raw_target_ids], dim=1)
            decoder_attention_mask = (decoder_input != self.pad_idx).long()

            # MODEL PASS
            # auto-regressive mask is handled internally by T5Model
            outputs = self.backbone(
                input_ids=encoder_input,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input,
                decoder_attention_mask=decoder_attention_mask,
            )

            # LOSS CALCULATION
            hidden_states = outputs.last_hidden_state
            logits = self.output_projection(hidden_states)

            # align logits and labels for loss computation
            # shift logits to exclude last time step for where there is no next token
            prediction_logits = logits[:, :-1, :].contiguous()

            unreduced_loss = self.loss_fct(
                prediction_logits.view(-1, self.output_vocab_size),
                raw_target_ids.view(-1)
            )
            
            # reshape back to [Batch, SeqLen]
            unreduced_loss = unreduced_loss.view(batch_size, -1)
            
            # sum over the sequence (dim=1), mean over the batch (dim=0)
            loss = unreduced_loss.sum(dim=1).mean()

            return {'loss': loss, 'logits': logits, 'unreduced_loss': unreduced_loss}
        
        else:
            # INFERENCE MODE
            # return encoder outputs for external generation loop (e.g. Beam Search)
            outputs = self.backbone.encoder(
                input_ids=encoder_input,
                attention_mask=encoder_attention_mask
            )
            return {
                'encoder_last_hidden_state': outputs.last_hidden_state,
                'encoder_attention_mask': encoder_attention_mask
            }

    @torch.no_grad()
    def beam_search(self, history_tuples, user_ids=None, beam_size=10, constrained=True):
        """
        Generates candidates using beam search.
        If constrained=True and codebooks are set, applies hierarchy and prefix constraints using the Trie index.
        
        :param history_tuples: [batch, T, codebook_layers]
        :param user_ids: [batch]
        :param beam_size: int
        :param constrained: bool, whether to apply Trie constraints
        :return: [batch, beam_size, codebook_layers + 1]
        """
        device = history_tuples.device
        batch_size = history_tuples.size(0)
        
        # ENCODE the user history once
        encoder_out = self.forward(history_tuples, target_tuples=None, user_ids=user_ids)
        encoder_hidden_states = encoder_out['encoder_last_hidden_state']
        encoder_attention_mask = encoder_out['encoder_attention_mask']

        # duplicate encoder outputs for each beam
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(beam_size, dim=0)
        encoder_attention_mask = encoder_attention_mask.repeat_interleave(beam_size, dim=0)

        # INITIALIZE BEAM SEARCH VARIABLES
        # every beam starts with BOS token
        current_sequences = torch.full((batch_size * beam_size, 1), self.bos_idx, dtype=torch.long, device=device)
        
        # initialize beam scores with -inf except for the first beam of each batch which starts at 0
        # this ensures that at the first step, only the first beam is expanded, thus avoiding duplicates of the same sequence
        beam_scores = torch.full((batch_size * beam_size,), -1e9, device=device)
        beam_scores[::beam_size] = 0

        # TRIE STATE INITIALIZATION
        # start at root (node 0) for all beams
        current_trie_nodes = torch.zeros(batch_size * beam_size, dtype=torch.long, device=device)

        # KV-CACHE INITIALIZATION
        past_key_values = None

        # GENERATION LOOP
        if self.codebooks is not None:
            total_steps = self.codebooks.shape[1]
        else:
            if constrained:
                logger.warning("Codebooks not set! Running unconstrained beam search.")
                constrained = False
            total_steps = self.codebook_layers + 1
        
        for step in range(total_steps):
            # prepare input
            # only use the last token if we have cached past_key_values
            if past_key_values is not None:
                decoder_input_ids = current_sequences[:, -1:]
            else:
                decoder_input_ids = current_sequences

            outputs = self.backbone.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # update cache
            past_key_values = outputs.past_key_values

            next_token_logits = self.output_projection(outputs.last_hidden_state[:, -1, :]) # shape: [B*K, Vocab]
            
            # LOG PROBS AND CONSTRAINED DECODING (TRIE LOOKUP)
            # compute raw log probabilities first to avoid NaN from purely -inf inputs
            log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
            
            if constrained and self.trie_masks is not None:
                # get valid token mask for current state of each beam (boolean mask)
                valid_token_mask = self.trie_masks[current_trie_nodes] # shape: [B*K, Vocab]
                
                # mask invalid paths with -inf
                log_probs = torch.where(valid_token_mask, log_probs, float('-inf'))

            # BEAM EXPANSION
            candidate_scores = beam_scores.unsqueeze(1) + log_probs # [B*K, Vocab]
            candidate_scores = candidate_scores.view(batch_size, -1) # [B, K*Vocab]
            
            # get top-k candidates across all beams and vocab
            topk_scores, topk_indices = torch.topk(candidate_scores, k=beam_size, dim=1)
            
            # resolve indices
            beam_indices = topk_indices // self.output_vocab_size # [B, K] - which beam in the group
            token_indices = topk_indices % self.output_vocab_size # [B, K] - which token
            
            # update scores
            beam_scores = topk_scores.view(-1) # [B*K]
            
            # UPDATE SEQUENCES & STATE
            # multiply batch offsets to get global beam indices
            batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * beam_size # [B, 1]
            global_beam_indices = (batch_offsets + beam_indices).view(-1) # [B*K]
            
            # select the sequences corresponding to the chosen beams
            selected_sequences = current_sequences[global_beam_indices] # [B*K, current_len]
            new_tokens = token_indices.view(-1, 1) # [B*K, 1]
            current_sequences = torch.cat([selected_sequences, new_tokens], dim=1)
            
            # update Trie state (transition to next node)
            if constrained and self.trie_transitions is not None:
                selected_nodes = current_trie_nodes[global_beam_indices] # [B*K]
                chosen_tokens_flat = token_indices.view(-1) # [B*K]
                
                # lookup next state
                # indices: [row=selected_nodes, col=chosen_tokens_flat]
                current_trie_nodes = self.trie_transitions[selected_nodes, chosen_tokens_flat].long()
            
            # REORDER CACHE
            # makes sure that the past_key_values are correctly aligned with the new beam order after selection
            if past_key_values is not None:
                if hasattr(self.backbone.decoder, '_reorder_cache'):
                    past_key_values = self.backbone.decoder._reorder_cache(past_key_values, global_beam_indices)
                elif hasattr(past_key_values, 'reorder_cache'):
                    past_key_values.reorder_cache(global_beam_indices)
            
        # remove BOS and reshape
        predictions = current_sequences[:, 1:].view(batch_size, beam_size, total_steps)
        
        return predictions

    @torch.no_grad()
    def generate(self, history_tuples, user_ids=None, num_items_to_generate=5, constrained=True):
        """
        Generates a sequence of recommended items using greedy decoding.
        If constrained=True, it follows the Trie to guarantee valid item IDs, 
        resetting state at item boundaries.

        :param history_tuples: A tensor of Semantic ID tuples representing the user's history.
        :param user_ids: An optional tensor of user IDs.
        :param num_items_to_generate: The number of complete item tuples to generate.
        :param constrained: bool, whether to apply Trie constraints.
        :return: A structured tensor of generated Semantic IDs. 
                 Shape: [Batch, num_items_to_generate, self.codebook_layers + 1].
        """
        device = history_tuples.device
        batch_size = history_tuples.size(0)

        # ENCODE STEP
        # call forward in inference mode (target=None)
        encoder_out = self.forward(history_tuples, target_tuples=None, user_ids=user_ids)
        encoder_hidden_states = encoder_out['encoder_last_hidden_state']
        encoder_attention_mask = encoder_out['encoder_attention_mask']

        # DECODE LOOP
        current_token = torch.full(
            (batch_size, 1), 
            self.bos_idx, 
            dtype=torch.long, 
            device=device
        )

        # initialize kv-cache
        past_key_values = None

        if self.codebooks is not None:
            tokens_per_item = self.codebooks.shape[1]
        else:
            if constrained:
                logger.warning("Codebooks not set! Running unconstrained generation.")
                constrained = False
            tokens_per_item = self.codebook_layers + 1
            
        total_tokens = num_items_to_generate * tokens_per_item

        generated_tokens = []
        
        # TRIE STATE INITIALIZATION for greedy decoding
        current_trie_nodes = torch.zeros(batch_size, dtype=torch.long, device=device)

        for step in range(total_tokens):
            # reset Trie at the beginning of each item
            if step > 0 and step % tokens_per_item == 0:
                current_trie_nodes = torch.zeros(batch_size, dtype=torch.long, device=device)

            outputs = self.backbone.decoder(
                input_ids=current_token,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )

            # update cache for the next step
            past_key_values = outputs.past_key_values

            # PREDICTION
            logits = self.output_projection(outputs.last_hidden_state[:, -1, :]) # [B, Vocab]
            
            if constrained and self.trie_masks is not None:
                valid_token_mask = self.trie_masks[current_trie_nodes]
                logits = torch.where(valid_token_mask, logits, float('-inf'))
                
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1) # [B, 1]
            generated_tokens.append(next_token)
            current_token = next_token
            
            # update Trie state
            if constrained and self.trie_transitions is not None:
                chosen_tokens_flat = next_token.view(-1) # [B]
                # indices: [row=current_trie_nodes, col=chosen_tokens_flat]
                current_trie_nodes = self.trie_transitions[current_trie_nodes, chosen_tokens_flat].long()

        # FORMAT OUTPUT
        flat_sequence = torch.cat(generated_tokens, dim=1) # [B, total_tokens]
        structured_output = flat_sequence.view(batch_size, num_items_to_generate, tokens_per_item) # [B, num_items, tokens_per_item]

        return structured_output