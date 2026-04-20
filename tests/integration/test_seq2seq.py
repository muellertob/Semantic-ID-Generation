"""
Integration tests for the TIGER Seq2Seq model (modules/recommender/seq2seq.py).

Covers:
  - Trie construction: node count, data types, sink node behaviour,
    root mask, shared prefix, leaf node masks, and edge cases
  - Constrained beam search: valid token forcing via prefix masks (all steps verified)
  - Dead beam handling: NaN prevention with sink nodes
  - Sequential multi-item generation: Trie state reset between items
  - KV-cache reordering: prefix branching with shared encoder state
  - Offset handling: beam search returns global IDs; metric fix verified
  - Forward pass: loss computation, logit shape, encoder-only mode
  - process_input_tuples: padding preservation, offset application, user token prepending
"""
import pytest
import torch
from modules.recommender import TigerSeq2Seq
from utils.metrics import MetricAccumulator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_model():
    """Minimal TigerSeq2Seq (codebook_layers=2, codebook_size=4) with tiny T5 backbone."""
    return TigerSeq2Seq(
        codebook_layers=2, codebook_size=4,
        d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2
    )


@pytest.fixture
def single_item_model(base_model):
    """base_model pre-loaded with one valid item [1, 2, 0] → shifted [1, 6, 8]."""
    base_model.set_codebooks(torch.tensor([[1, 2, 0]]))
    return base_model


@pytest.fixture
def two_item_model(base_model):
    """base_model pre-loaded with two items sharing layer-0 prefix: [1,2,0] and [1,3,0]."""
    base_model.set_codebooks(torch.tensor([[1, 2, 0], [1, 3, 0]]))
    return base_model


# ---------------------------------------------------------------------------
# TRIE CONSTRUCTION TESTS
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTrieConstruction:
    """
    Verifies that build_trie produces structurally correct transition and mask tables.

    Fixture uses codebook_layers=2, codebook_size=4 with three raw items:
      Item 0: [0, 1, 0]  →  shifted [0, 5, 8]
      Item 1: [0, 2, 0]  →  shifted [0, 6, 8]   (shares layer-0 prefix with item 0)
      Item 2: [3, 0, 0]  →  shifted [3, 4, 8]

    Expected trie (9 valid nodes + 1 sink = 10 total).
    Each children dict is read as {token → next_node}: seeing that token moves
    the state machine from the current node to the listed next node.

      Node 0 (root): children {0→1, 3→6}
      Node 1:        children {5→2, 6→4}   ← shared by items 0 and 1
      Node 2:        children {8→3}
      Node 3:        leaf  (item 0)
      Node 4:        children {8→5}
      Node 5:        leaf  (item 1)
      Node 6:        children {4→7}
      Node 7:        children {8→8}
      Node 8:        leaf  (item 2)
      Node 9:        sink  (traps all invalid transitions)
    """

    @pytest.fixture(scope="class")
    def model(self):
        m = TigerSeq2Seq(codebook_layers=2, codebook_size=4)
        m.set_codebooks(torch.tensor([
            [0, 1, 0],
            [0, 2, 0],
            [3, 0, 0],
        ]))
        return m

    # --- dtype / shape / sink ---

    def test_transition_tensor_is_int32(self, model):
        assert model.trie_transitions.dtype == torch.int32

    def test_mask_tensor_is_bool(self, model):
        assert model.trie_masks.dtype == torch.bool

    def test_total_node_count(self, model):
        assert model.trie_transitions.shape[0] == 10
        assert model.trie_masks.shape[0] == 10

    def test_sink_node_traps_all_transitions(self, model):
        sink = 9
        assert (model.trie_transitions[sink] == sink).all()

    def test_sink_node_mask_is_all_false(self, model):
        assert not model.trie_masks[9].any()

    # --- root node ---

    def test_root_allows_exactly_the_two_valid_first_tokens(self, model):
        """Root must allow shifted tokens 0 and 3 — and nothing else."""
        root_mask = model.trie_masks[0]
        assert root_mask[0].item(), "Root must allow shifted token 0 (items 0 & 1)"
        assert root_mask[3].item(), "Root must allow shifted token 3 (item 2)"
        assert root_mask.sum().item() == 2

    def test_root_routes_token_0_to_valid_node(self, model):
        assert model.trie_transitions[0, 0].item() != 9

    def test_root_routes_token_3_to_valid_node(self, model):
        assert model.trie_transitions[0, 3].item() != 9

    def test_root_routes_unregistered_token_to_sink(self, model):
        # Token 1 is not a valid layer-0 first token in this fixture
        assert model.trie_transitions[0, 1].item() == 9

    # --- shared prefix ---

    def test_shared_prefix_routes_both_items_through_same_node(self, model):
        """Items [0,5,8] and [0,6,8] must share the single node reached via root→0."""
        shared = model.trie_transitions[0, 0].item()
        masks = model.trie_masks
        assert masks[shared, 5].item(), "Shared node must allow token 5 (item 0 next step)"
        assert masks[shared, 6].item(), "Shared node must allow token 6 (item 1 next step)"
        assert masks[shared].sum().item() == 2

    def test_shared_prefix_node_rejects_layer0_tokens(self, model):
        """After descending one level, root-layer tokens must not appear as valid."""
        shared = model.trie_transitions[0, 0].item()
        masks = model.trie_masks
        assert not masks[shared, 0].item()
        assert not masks[shared, 3].item()

    # --- leaf nodes ---

    def test_leaf_nodes_have_empty_masks(self, model):
        t = model.trie_transitions
        masks = model.trie_masks

        def reach_leaf(path):
            node = 0
            for tok in path:
                node = t[node, tok].item()
            return node

        for shifted_path in [[0, 5, 8], [0, 6, 8], [3, 4, 8]]:
            leaf = reach_leaf(shifted_path)
            assert not masks[leaf].any(), f"Leaf at {shifted_path} must have empty mask"

    # --- full reachability ---

    def test_all_valid_paths_are_fully_reachable(self, model):
        """Every shifted item must be traversable end-to-end without hitting the sink."""
        t = model.trie_transitions
        sink = 9
        for path in [[0, 5, 8], [0, 6, 8], [3, 4, 8]]:
            node = 0
            for step, tok in enumerate(path):
                next_node = t[node, tok].item()
                assert next_node != sink, f"Path {path} hit sink at step {step} (token {tok})"
                node = next_node


# --- Trie edge cases ---

@pytest.mark.integration
def test_trie_single_item_creates_minimal_structure():
    m = TigerSeq2Seq(codebook_layers=2, codebook_size=4)
    m.set_codebooks(torch.tensor([[0, 0, 0]]))  # shifted → [0, 4, 8]

    # 1 root + 3 path nodes + 1 sink = 5
    assert m.trie_transitions.shape[0] == 5
    assert m.trie_masks[0, 0].item()
    assert m.trie_masks[0].sum().item() == 1


@pytest.mark.integration
def test_trie_duplicate_items_do_not_inflate_node_count():
    m = TigerSeq2Seq(codebook_layers=2, codebook_size=4)
    m.set_codebooks(torch.tensor([[0, 0, 0], [0, 0, 0]]))

    assert m.trie_transitions.shape[0] == 5


@pytest.mark.integration
def test_trie_out_of_vocab_tokens_are_silently_dropped():
    """Tokens >= output_vocab_size must be excluded from transition tables without error."""
    m = TigerSeq2Seq(codebook_layers=2, codebook_size=4)  # output_vocab_size = 15

    # call build_trie directly with a pre-shifted sequence whose last token is out of vocab.
    # in normal usage set_codebooks() produces in-range tokens; this tests the guard at
    # build_trie line: `if token < self.output_vocab_size`.
    bogus_shifted = torch.tensor([[0, 4, 99]])  # 99 >= 15 → must be skipped
    m.build_trie(bogus_shifted)

    t = m.trie_transitions
    masks = m.trie_masks

    # in-vocab tokens still create valid transitions
    assert masks[0, 0].item(), "Root must allow in-vocab token 0"
    node_after_0 = t[0, 0].item()
    assert masks[node_after_0, 4].item(), "Depth-1 node must allow in-vocab token 4"

    # node reached after (0 → 4) has no further mask entry because 99 was dropped
    node_after_4 = t[node_after_0, 4].item()
    assert not masks[node_after_4].any(), (
        "Depth-2 node must have empty mask: out-of-vocab token 99 was silently dropped"
    )


# ---------------------------------------------------------------------------
# BEAM SEARCH TESTS
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_beam_search_constrained_forces_valid_path(single_item_model):
    """Constrained beam search must follow the trie even when a high-logit invalid token exists."""
    model = single_item_model
    history = torch.tensor([[[-1, -1, -1]]])

    class MockOutputProjection(torch.nn.Module):
        def forward(self, hidden_states):
            vocab_size = model.output_vocab_size
            logits = torch.zeros((hidden_states.shape[0], vocab_size), device=hidden_states.device)
            logits[:, 0] = 100.0    # invalid token — highest score
            logits[:, 1] = -100.0   # valid layer-0 token
            logits[:, 6] = -100.0   # valid layer-1 token
            logits[:, 8] = -100.0   # valid layer-2 token
            return logits

    model.output_projection = MockOutputProjection()

    preds = model.beam_search(history, beam_size=1, constrained=True)
    assert preds[0, 0, 0].item() == 1, "Constrained failed at step 0"
    assert preds[0, 0, 1].item() == 6, "Constrained failed at step 1"
    assert preds[0, 0, 2].item() == 8, "Constrained failed at step 2"


@pytest.mark.integration
def test_beam_search_unconstrained_picks_highest_logit(single_item_model):
    """Unconstrained beam search must greedily follow the highest-scoring token at each step."""
    model = single_item_model
    history = torch.tensor([[[-1, -1, -1]]])

    class MockOutputProjection(torch.nn.Module):
        def forward(self, hidden_states):
            vocab_size = model.output_vocab_size
            logits = torch.zeros((hidden_states.shape[0], vocab_size), device=hidden_states.device)
            logits[:, 0] = 100.0   # dominates every step
            return logits

    model.output_projection = MockOutputProjection()

    preds = model.beam_search(history, beam_size=1, constrained=False)
    assert preds[0, 0, 0].item() == 0
    assert preds[0, 0, 1].item() == 0
    assert preds[0, 0, 2].item() == 0


@pytest.mark.integration
def test_beam_search_dead_beam_handling(single_item_model):
    """With more beams than valid items, all beams must converge to the only valid path.

    Dead beams (score = -1e9) still obey the trie constraint; since the only
    valid token at each node is the one on the single valid path, topk selects
    -1e9 over -inf and all beams naturally follow that path.
    """
    model = single_item_model
    history = torch.tensor([[[-1, -1, -1]]])
    preds = model.beam_search(history, beam_size=3, constrained=True)

    # shape must always be (batch, beam_size, tokens_per_item)
    assert preds.shape == (1, 3, 3)

    # no beam may contain NaN — dead beams must degrade gracefully
    assert not torch.isnan(preds.float()).any()

    # with only 1 valid item, every beam must fall back to that item
    for beam in range(3):
        assert preds[0, beam, 0].item() == 1, f"Beam {beam} step 0 must be the valid item"
        assert preds[0, beam, 1].item() == 6, f"Beam {beam} step 1 must be the valid item"
        assert preds[0, beam, 2].item() == 8, f"Beam {beam} step 2 must be the valid item"


@pytest.mark.integration
@pytest.mark.parametrize("beam_size", [1, 3, 5])
def test_beam_search_output_shape_matches_beam_size(beam_size):
    """Output shape must always be (batch, beam_size, tokens_per_item)."""
    m = TigerSeq2Seq(codebook_layers=2, codebook_size=4)
    m.set_codebooks(torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]]))
    history = torch.tensor([[[-1, -1, -1]]])
    preds = m.beam_search(history, beam_size=beam_size, constrained=True)
    assert preds.shape == (1, beam_size, 3)


@pytest.mark.integration
def test_beam_search_ranks_items_by_log_probability():
    """The beam with the higher cumulative log-probability must come first."""
    m = TigerSeq2Seq(codebook_layers=2, codebook_size=4)
    # shifted: item A → [0, 4, 8], item B → [1, 4, 8]
    m.set_codebooks(torch.tensor([[0, 0, 0], [1, 0, 0]]))
    history = torch.tensor([[[-1, -1, -1]]])

    class LogitsByToken(torch.nn.Module):
        def forward(self, hidden_states):
            logits = torch.full((hidden_states.shape[0], m.output_vocab_size), -100.0)
            logits[:, 0] = 90.0   # item A's first token — higher logit
            logits[:, 1] = 10.0   # item B's first token — lower logit
            logits[:, 4] = 50.0   # shared layer-1 token
            logits[:, 8] = 50.0   # shared layer-2 token
            return logits

    m.output_projection = LogitsByToken()
    preds = m.beam_search(history, beam_size=2, constrained=True)

    assert preds[0, 0, 0].item() == 0, "Higher-logit item A must be beam 0"
    assert preds[0, 1, 0].item() == 1, "Lower-logit item B must be beam 1"


@pytest.mark.integration
def test_beam_search_batch_size_two_produces_correct_shape():
    """beam_search must handle batch_size > 1 and produce independent results per item."""
    m = TigerSeq2Seq(codebook_layers=2, codebook_size=4)
    m.set_codebooks(torch.tensor([[0, 0, 0], [1, 0, 0]]))
    history = torch.tensor([[[-1, -1, -1]], [[-1, -1, -1]]])  # batch_size=2
    preds = m.beam_search(history, beam_size=2, constrained=True)
    assert preds.shape == (2, 2, 3)
    # identical inputs → same candidate set per batch item; beam ordering may differ due to
    # floating-point non-determinism across batch positions, so compare as sets of tuples
    set0 = {tuple(x.tolist()) for x in preds[0]}
    set1 = {tuple(x.tolist()) for x in preds[1]}
    assert set0 == set1, "Different batch positions with identical input produced different candidate sets"


@pytest.mark.integration
def test_beam_search_unconstrained_falls_back_without_codebooks():
    """beam_search without set_codebooks must run unconstrained using codebook_layers + 1 steps."""
    m = TigerSeq2Seq(codebook_layers=2, codebook_size=4)
    history = torch.tensor([[[-1, -1, -1]]])
    preds = m.beam_search(history, beam_size=1, constrained=False)
    assert preds.shape == (1, 1, 3)


@pytest.mark.integration
def test_generate_sequential_items_constrained(single_item_model):
    """Trie state must reset between items; requesting 3 items must not break on item 2+."""
    model = single_item_model
    history = torch.tensor([[[-1, -1, -1]]])
    preds = model.generate(history, num_items_to_generate=3, constrained=True)

    assert preds.shape == (1, 3, 3)
    for i in range(3):
        assert preds[0, i, 0].item() == 1, f"Item {i} step 0 invalid"
        assert preds[0, i, 1].item() == 6, f"Item {i} step 1 invalid"
        assert preds[0, i, 2].item() == 8, f"Item {i} step 2 invalid"


@pytest.mark.integration
def test_beam_search_cache_reordering(two_item_model):
    """Two items sharing a prefix force cache duplication and reordering."""
    model = two_item_model
    history = torch.tensor([[[-1, -1, -1]]])

    class MockPrefixBranchingProjection(torch.nn.Module):
        def forward(self, hidden_states):
            vocab_size = model.output_vocab_size
            logits = torch.zeros((hidden_states.shape[0], vocab_size), device=hidden_states.device) - 100.0
            logits[:, 1] = 100.0   # layer-0 token 1 wins easily
            logits[:, 6] = 90.0    # layer-1 token 2 (higher)
            logits[:, 7] = 80.0    # layer-1 token 3 (lower)
            logits[:, 8] = 100.0   # layer-2 token 0 must win
            return logits

    model.output_projection = MockPrefixBranchingProjection()

    preds = model.beam_search(history, beam_size=2, constrained=True)

    assert preds[0, 0, 0].item() == 1
    assert preds[0, 0, 1].item() == 6
    assert preds[0, 0, 2].item() == 8

    assert preds[0, 1, 0].item() == 1
    assert preds[0, 1, 1].item() == 7
    assert preds[0, 1, 2].item() == 8


# ---------------------------------------------------------------------------
# FORWARD PASS TESTS
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestForwardPass:
    """Verifies TigerSeq2Seq.forward() for training (with targets) and inference (without)."""

    @pytest.fixture(scope="class")
    def model(self):
        m = TigerSeq2Seq(
            codebook_layers=2, codebook_size=4,
            d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2
        )
        m.eval()
        return m

    @pytest.fixture
    def history(self):
        # batch=2, seq_len=3, tokens_per_tuple=3
        return torch.tensor([
            [[0, 1, 0], [1, 0, 2], [-1, -1, -1]],
            [[2, 3, 1], [-1, -1, -1], [-1, -1, -1]],
        ])

    @pytest.fixture
    def target(self):
        return torch.tensor([[[1, 2, 0]], [[0, 1, 3]]])  # [batch=2, 1, tuple_size=3]

    def test_forward_with_target_returns_loss(self, model, history, target):
        out = model(history, target_tuples=target)
        assert "loss" in out

    def test_forward_loss_is_positive(self, model, history, target):
        out = model(history, target_tuples=target)
        assert out["loss"].item() > 0.0

    def test_forward_loss_is_differentiable(self, model, history, target):
        model.train()
        out = model(history, target_tuples=target)
        out["loss"].backward()
        model.eval()

    def test_forward_logits_vocab_dimension(self, model, history, target):
        """Logit tensor last dim must equal output_vocab_size."""
        out = model(history, target_tuples=target)
        assert out["logits"].shape[-1] == model.output_vocab_size

    def test_forward_logits_batch_dimension(self, model, history, target):
        """Logit tensor first dim must equal batch size."""
        out = model(history, target_tuples=target)
        assert out["logits"].shape[0] == history.shape[0]

    def test_forward_without_target_returns_encoder_states(self, model, history):
        out = model(history)
        assert "encoder_last_hidden_state" in out
        assert "encoder_attention_mask" in out

    def test_forward_encoder_attention_mask_zeros_for_padding(self, model):
        """Padding tokens (-1) in the history must produce zeros in the attention mask."""
        all_padding = torch.full((1, 2, 3), -1, dtype=torch.long)
        out = model(all_padding)
        assert out["encoder_attention_mask"].sum().item() == 0


# ---------------------------------------------------------------------------
# PROCESS_INPUT_TUPLES TESTS
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestProcessInputTuples:
    """Verifies that process_input_tuples correctly handles padding, offsets, and user tokens."""

    @pytest.fixture(scope="class")
    def model(self):
        return TigerSeq2Seq(
            codebook_layers=2, codebook_size=4,
            d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2
        )

    def test_applies_offsets_to_non_padding_tokens(self, model):
        """Each token position k must be shifted by item_offsets[k]."""
        # single item, no padding: raw [1, 2, 0] — offsets are [0, 4, 8]
        tuples = torch.tensor([[[1, 2, 0]]])  # [batch=1, seq=1, tuple=3]
        out = model.process_input_tuples(tuples)
        expected = torch.tensor([[1, 2 + 4, 0 + 8]])
        assert torch.equal(out, expected)

    def test_preserves_padding_as_pad_idx(self, model):
        """Positions with raw value -1 must appear as pad_idx in the output."""
        tuples = torch.tensor([[[-1, -1, -1]]])  # all padding
        out = model.process_input_tuples(tuples)
        assert (out == model.pad_idx).all()

    def test_mixed_padding_and_real_tokens(self, model):
        """Real tokens in a sequence are shifted; padding positions remain pad_idx."""
        tuples = torch.tensor([[[1, 0, 0], [-1, -1, -1]]])
        out = model.process_input_tuples(tuples)  # flat [batch, 6]
        assert out[0, 0].item() == 1   # 1 + offset[0]=0
        assert out[0, 1].item() == 4   # 0 + offset[1]=4
        assert out[0, 2].item() == 8   # 0 + offset[2]=8
        assert (out[0, 3:] == model.pad_idx).all()

    def test_output_shape_without_user_ids(self, model):
        """Output must be [batch, seq_len * tuple_size] when no user IDs are provided."""
        tuples = torch.zeros((4, 5, 3), dtype=torch.long)
        out = model.process_input_tuples(tuples)
        assert out.shape == (4, 5 * 3)

    def test_output_shape_with_user_ids(self, model):
        """Output must be [batch, 1 + seq_len * tuple_size] when user IDs are provided."""
        tuples = torch.zeros((4, 5, 3), dtype=torch.long)
        user_ids = torch.arange(4)
        out = model.process_input_tuples(tuples, user_ids=user_ids)
        assert out.shape == (4, 1 + 5 * 3)

    def test_user_token_is_prepended_within_valid_range(self, model):
        """First token in the output must be within the user token range."""
        tuples = torch.zeros((1, 1, 3), dtype=torch.long)
        user_ids = torch.tensor([0])
        out = model.process_input_tuples(tuples, user_ids=user_ids)
        user_token = out[0, 0].item()
        assert user_token >= model.user_offset
        assert user_token < model.user_offset + model.user_vocab_size


# ---------------------------------------------------------------------------
# OFFSET HANDLING TESTS
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestOffsetHandling:
    """
    Verifies item_offsets correctness and the offset–metric relationship.

    With codebook_layers=3, codebook_size=10:
      item_offsets: [0, 10, 20, 30]
        - layer k offset = k * codebook_size
        - collision layer offset = codebook_layers * codebook_size = semantic_vocab_size
        - layer 0 is unshifted (offset = 0): raw and global IDs are identical

    Metric tests: MetricAccumulator compares raw IDs, so beam-search predictions
    (which carry global IDs) must have item_offsets subtracted before evaluation.

    Note: tests that mutate model state via set_codebooks() live as standalone
    functions below so the shared scope="class" model remains unmodified.
    """

    @pytest.fixture(scope="class")
    def model(self):
        m = TigerSeq2Seq(codebook_layers=3, codebook_size=10, d_model=32, num_layers=2, num_heads=4)
        m.eval()
        return m

    # --- item_offsets shape and values ---

    def test_item_offsets_shape(self, model):
        """item_offsets must have one entry per token position (codebook_layers + 1)."""
        assert model.item_offsets.shape == (model.codebook_layers + 1,)

    def test_item_offsets_values(self, model):
        """Each layer k offset = k * codebook_size; collision offset = codebook_layers * codebook_size."""
        assert model.item_offsets.tolist() == [0, 10, 20, 30]

    def test_layer_zero_has_zero_offset(self, model):
        """Layer-0 raw and global IDs are identical (offset = 0)."""
        assert model.item_offsets[0].item() == 0

    def test_collision_offset_equals_semantic_vocab_size(self, model):
        """The collision-layer offset must equal semantic_vocab_size."""
        assert model.item_offsets[-1].item() == model.semantic_vocab_size

    # --- metric interaction ---

    def test_raw_vs_global_id_mismatch_produces_zero_recall(self, model):
        """Global IDs compared against raw targets → tuple mismatch → recall = 0."""
        acc = MetricAccumulator(k_list=[1], num_layers=model.codebook_layers + 1)

        raw = torch.tensor([1, 1, 1, 0])
        target_raw = raw.view(1, 1, -1)
        prediction_global = (raw + model.item_offsets).view(1, 1, -1)  # [1, 11, 21, 30]

        acc.update(prediction_global, target_raw)
        assert acc.compute()["recall"][1] == 0.0

    def test_offset_correction_restores_recall_to_one(self, model):
        """Subtracting item_offsets from global predictions restores recall to 1.0."""
        acc = MetricAccumulator(k_list=[1], num_layers=model.codebook_layers + 1)

        raw = torch.tensor([1, 1, 1, 0])
        target_raw = raw.view(1, 1, -1)
        prediction_global = (raw + model.item_offsets).view(1, 1, -1)
        prediction_raw = prediction_global - model.item_offsets.view(1, 1, -1)

        acc.update(prediction_raw, target_raw)
        assert acc.compute()["recall"][1] == 1.0

    def test_single_wrong_layer_offset_produces_zero_recall(self, model):
        """An off-by-one error in any one layer's offset collapses recall to 0."""
        raw = torch.tensor([1, 1, 1, 0])
        target_raw = raw.view(1, 1, -1)

        wrong_offsets = model.item_offsets.clone()
        wrong_offsets[2] += 1  # corrupt layer 2: [0, 10, 21, 30] instead of [0, 10, 20, 30]
        prediction = (raw + wrong_offsets).view(1, 1, -1)

        acc = MetricAccumulator(k_list=[1], num_layers=model.codebook_layers + 1)
        acc.update(prediction, target_raw)
        assert acc.compute()["recall"][1] == 0.0


@pytest.mark.integration
def test_beam_search_returns_global_ids_for_all_layers():
    """beam_search output must carry the correct offset for every token position.

    With an all-zero raw codebook, the expected global output is item_offsets
    themselves: [0, 10, 20, 30]. Checking only one layer (as the prior test did)
    misses that layer-0 is trivially 0 and layers 2/3 are not shifted correctly.
    """
    m = TigerSeq2Seq(codebook_layers=3, codebook_size=10, d_model=32, num_layers=2, num_heads=4)
    m.eval()

    # all-zero raw codebook: after set_codebooks offset → [0, 10, 20, 30]
    m.set_codebooks(torch.zeros((1, 4), dtype=torch.long))
    history = torch.zeros((1, 5, 4), dtype=torch.long)

    with torch.no_grad():
        preds = m.beam_search(history, beam_size=1, constrained=True)

    assert preds.shape == (1, 1, 4)
    expected = m.item_offsets.tolist()  # [0, 10, 20, 30]
    actual = preds[0, 0].tolist()
    assert actual == expected, f"Expected global IDs {expected}, got {actual}"


@pytest.mark.integration
def test_beam_search_offset_subtraction_restores_raw_ids():
    """Subtracting item_offsets from beam search output recovers the original raw codebook IDs."""
    m = TigerSeq2Seq(codebook_layers=3, codebook_size=10, d_model=32, num_layers=2, num_heads=4)
    m.eval()

    raw_codebook = torch.tensor([[3, 5, 7, 2]], dtype=torch.long)  # distinct values per layer
    m.set_codebooks(raw_codebook)
    history = torch.zeros((1, 5, 4), dtype=torch.long)

    with torch.no_grad():
        preds = m.beam_search(history, beam_size=1, constrained=True)

    raw_preds = preds[0, 0] - m.item_offsets

    assert (raw_preds >= 0).all(), "All recovered raw IDs must be non-negative"
    assert (raw_preds < m.codebook_size).all(), "All recovered raw IDs must be < codebook_size"
    assert raw_preds.tolist() == raw_codebook[0].tolist(), \
        "Recovered raw IDs must exactly match the input codebook"
