import torch
import torch.nn as nn
import logging
from transformers import T5Config, T5Model

logger = logging.getLogger(__name__)

class TigerSeq2Seq(nn.Module):
    def __init__(self, codebook_layers=3, codebook_size=256, user_tokens=2000, d_model=512, num_layers=4, num_heads=6, dropout=0.1, activation_fn="relu"):
        super().__init__()
        self.codebook_layers = codebook_layers
        self.codebook_size = codebook_size

        # VOCABULARY SIZE
        self.semantic_vocab_size = codebook_layers * codebook_size
        self.collision_vocab_size = codebook_size # additional tokens for ID collisions
        self.user_vocab_size = user_tokens

        self.total_vocab = (self.semantic_vocab_size + 
                            self.collision_vocab_size + 
                            self.user_vocab_size + 
                            3)
        
        # set special token indices
        self.pad_idx = self.total_vocab - 3
        self.bos_idx = self.total_vocab - 2
        self.eos_idx = self.total_vocab - 1

        # OFFSETS: shift tokens to avoid overlaps in the shared embedding space
        semantic_offsets = torch.arange(0, codebook_layers * codebook_size, step=codebook_size)
        collision_start = torch.tensor([self.semantic_vocab_size])

        # register as buffer to move with the model to device
        self.register_buffer('item_offsets', 
                             torch.cat([semantic_offsets, collision_start]))
        
        # collision tokens start after semantic tokens
        self.collision_offset = self.semantic_vocab_size

        # user tokens start after semantic + collision tokens
        self.user_offset = self.semantic_vocab_size + self.collision_vocab_size

        # LOSS FUNCTION
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        # TRANSFORMER MODEL
        # using T5 as the backbone seq2seq model like in the original TIGER paper
        config = T5Config(
            vocab_size=self.total_vocab,
            d_model=d_model,                    # TIGER: 128 (Input Dimension)
            d_kv=d_model // 8,                  # TIGER: 64
            d_ff=d_model * 4,                   # TIGER: 1024 (MLP Dimension)
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

        # OUTPUT PROJECTION: maps hidden state back to vocab probabilities
        self.output_projection = nn.Linear(d_model, self.total_vocab, bias=False)
        
        # codebook buffer for constrained generation
        # will be set via set_codebooks()
        self.register_buffer('codebooks', None)

    def set_codebooks(self, codebooks):
        """
        Registers the valid Semantic IDs for constrained generation.
        Applies offsets to match the model's vocabulary space.
        
        :param codebooks: Tensor of shape [Num_Items, Codebook_Layers + 1] containing raw indices.
        """
        device = self.item_offsets.device
        if codebooks.device != device:
            codebooks = codebooks.to(device)
            
        # ensure dimensionality matches for broadcasting
        # codebooks: [N, L]; item_offsets: [L]
        if codebooks.shape[1] != self.item_offsets.shape[0]:
             raise ValueError(f"Codebook shape {codebooks.shape} does not match item offsets shape {self.item_offsets.shape}")
        
        # apply offsets to raw codebook indices to match vocabulary   
        shifted_codebooks = codebooks + self.item_offsets.unsqueeze(0)
        self.register_buffer('codebooks', shifted_codebooks)

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

            loss = self.loss_fct(
                prediction_logits.view(-1, self.total_vocab),
                raw_target_ids.view(-1)
            )

            return {'loss': loss, 'logits': logits}
        
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

    def _check_valid_prefix(self, prefix, batch_size=10000):
        """
        Checks if a given prefix is a valid prefix of the codebooks.
        
        Args:
            prefix: Tensor of shape [B, Hierarchy_Level] containing shifted token IDs.
        Returns:
            Boolean Tensor of shape [B]
        """
        # if no codebooks are set, we consider all prefixes as valid (unconstrained generation)
        if self.codebooks is None:
            return torch.ones(prefix.shape[0], dtype=torch.bool, device=prefix.device)
            
        current_hierarchy = prefix.shape[1]
        num_prefixes = prefix.shape[0]
        results = []
        target_codebooks = self.codebooks
        
        # trim to current hierarchy depth
        trimmed_codebooks = target_codebooks[:, :current_hierarchy]
        
        # process in batches to avoid OOM
        for i in range(0, num_prefixes, batch_size):
            batch_prefix = prefix[i:i+batch_size]
            
            # broadcasting comparison:
            # batch_prefix: [B_chunk, H] -> [1, B_chunk, H]
            # trimmed_codebooks: [N, H] -> [N, 1, H]
            # result: [N, B_chunk, H]
            comparison = (trimmed_codebooks.unsqueeze(1) == batch_prefix.unsqueeze(0))
            
            # check if ALL tokens in the prefix match a codebook entry
            # [N, B_chunk]
            match_per_codebook = comparison.all(dim=2)
            
            # check if ANY codebook entry matches this prefix
            # [B_chunk]
            is_valid = match_per_codebook.any(dim=0)
            
            results.append(is_valid)
            
        return torch.cat(results)

    # CONTINUE HERE
    @torch.no_grad()
    def beam_search(self, history_tuples, user_ids=None, beam_size=10):
        """
        Generates candidates using (constrained) beam search.
        If codebooks are set, applies hierarchy and prefix constraints.
        
        :param history_tuples: [batch, T, codebook_layers]
        :param user_ids: [batch]
        :param beam_size: int
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
        # first token is BOS, so initial score is 0 for the first token of each beam, and -inf for the rest
        # this ensures that at the first step, only the first beam is expanded, thus avoiding duplicates of the same sequence
        beam_scores = torch.full((batch_size * beam_size,), -1e9, device=device)
        beam_scores[::beam_size] = 0

        # GENERATION LOOP
        if self.codebooks is not None:
            total_steps = self.codebooks.shape[1]
        else:
            logger.warning("Codebooks not set! Running unconstrained beam search. Generated IDs may not exist.")
            total_steps = self.codebook_layers + 1
        
        for step in range(total_steps):
            outputs = self.backbone.decoder(
                input_ids=current_sequences,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )
            
            next_token_logits = self.output_projection(outputs.last_hidden_state[:, -1, :]) # shape: [B*K, Vocab]
            
            # OFFSET MASKING: enforce hierarchy by masking out invalid token ranges at each step
            # step i -> tokens [offset[i], offset[i+1])
            
            # calculate start and end indices for valid tokens at this step
            if step < self.codebook_layers:
                 start_idx = self.item_offsets[step]
                 end_idx = self.item_offsets[step+1]
            else:
                 # collision Step (last step)
                 start_idx = self.item_offsets[-1]
                 end_idx = start_idx + self.collision_vocab_size
            
            # create mask to set invalid token logits to -inf
            vocab_mask = torch.full_like(next_token_logits, float('-inf'))
            vocab_mask[:, start_idx:end_idx] = 0
            
            # APPLY PREFIX CONSTRAINTS (if codebooks are set)
            # check if appending a candidate token creates a valid prefix
            if self.codebooks is not None:
                # get all candidate tokens for the current step (those within the vocab range)
                candidate_tokens = torch.arange(start_idx, end_idx, device=device)
                
                # get history for each beam (exclude BOS)
                # first step -> empty history with shape [B*K, 0]
                history_no_bos = current_sequences[:, 1:] # [B*K, step]
                
                # repeat history for each candidate
                expanded_history = history_no_bos.repeat_interleave(candidate_tokens.shape[0], dim=0) # [B*K * candidate_tokens, step]
                
                # create matching column of candidate tokens to append to expanded history
                tiled_candidates = candidate_tokens.repeat(history_no_bos.shape[0]).unsqueeze(1) # [B*K * candidate_tokens, 1]
                
                # concatenate to form new prefixes to check
                prefixes_to_check = torch.cat([expanded_history, tiled_candidates], dim=1)
                
                # check validity
                is_valid = self._check_valid_prefix(prefixes_to_check) # [B*K * candidate_tokens]
                
                # reshape back to [B*K, candidate_tokens]
                validity_mask = is_valid.view(history_no_bos.shape[0], candidate_tokens.shape[0])
                
                # APPLY THE VALIDITY MASK
                # create a local mask for the constrained range
                local_mask = torch.full_like(validity_mask, float('-inf'), dtype=next_token_logits.dtype)
                local_mask[validity_mask] = 0
                
                # add to the global vocab mask slice
                vocab_mask[:, start_idx:end_idx] += local_mask

            # apply mask and calculate candidate scores
            log_probs = torch.nn.functional.log_softmax(next_token_logits + vocab_mask, dim=-1)
            
            candidate_scores = beam_scores.unsqueeze(1) + log_probs # [B*K, Vocab]
            candidate_scores = candidate_scores.view(batch_size, -1) # [B, K*Vocab]
            
            # get top-k candidates across all beams and vocab
            topk_scores, topk_indices = torch.topk(candidate_scores, k=beam_size, dim=1)
            
            # resolve indices
            beam_indices = topk_indices // self.total_vocab # [B, K] - which beam in the group
            token_indices = topk_indices % self.total_vocab # [B, K] - which token
            
            # update scores
            beam_scores = topk_scores.view(-1) # [B*K]
            
            # UPDATE SEQUENCES
            # multiply batch offsets to get global beam indices
            batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * beam_size # [B, 1]
            global_beam_indices = (batch_offsets + beam_indices).view(-1) # [B*K]
            
            # select the sequences corresponding to the chosen beams and append the new tokens
            selected_sequences = current_sequences[global_beam_indices] # [B*K, current_len]
            new_tokens = token_indices.view(-1, 1) # [B*K, 1]
            current_sequences = torch.cat([selected_sequences, new_tokens], dim=1)
            
        # remove BOS and reshape
        predictions = current_sequences[:, 1:].view(batch_size, beam_size, total_steps)
        
        return predictions

    @torch.no_grad()
    def generate(self, history_tuples, user_ids=None, num_items_to_generate=5):
        """
        Generates a sequence of recommended items using greedy decoding.

        :param history_tuples: A tensor of Semantic ID tuples representing the user's history.
        :param user_ids: An optional tensor of user IDs.
        :param num_items_to_generate: The number of complete item tuples to generate.
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

        # initialize cache
        past_key_values = None

        tokens_per_item = self.codebook_layers + 1
        total_tokens = num_items_to_generate * tokens_per_item

        generated_tokens = []

        for _ in range(total_tokens):
            outputs = self.backbone.decoder(
                input_ids=current_token,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values, # cache the attention of previously generated tokens
                use_cache=True
            )

            # update cache for the next step
            past_key_values = outputs.past_key_values

            # PREDICTION
            logits = self.output_projection(outputs.last_hidden_state) # [B, 1, Vocab]
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1) # squeeze out the middle dimension

            generated_tokens.append(next_token)
            current_token = next_token

        # FORMAT OUTPUT
        flat_sequence = torch.cat(generated_tokens, dim=1) # [B, total_tokens]
        structured_output = flat_sequence.view(batch_size, num_items_to_generate, tokens_per_item) # [B, num_items, tokens_per_item]

        return structured_output