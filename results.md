# Results: layer-wise linear probes for next-token prediction in GPT-2

## Question
1. How readily can a linear probe decode the next token from each layer of a frozen GPT-2 small?
2. How does that layer-wise pattern differ between positions GPT-2 itself finds easy and positions it finds hard?

## Method
- **Model and probes:** one full-vocabulary linear probe (768 → 50,257) for each of 13 representations of a frozen GPT-2 small: the embedding output (layer 0) and the 12 block outputs, with layer 12 taken after the final LayerNorm. The input is the hidden state at position t; the label is the token at t+1.
- **Data:** WikiText-2 raw, split by article into 522 / 74 / 150 train / val / test articles before examples are built. From non-overlapping 256-token windows: 200k train, 20k val, 40k test positions.
- **Training:**
  - AdamW on standardised features, with one learning rate (3e-4) chosen by validation loss on layer 6.
  - Early stopping on validation loss, at most 10 epochs.
  - Identical examples and procedure for every layer.
- **Difficulty:** GPT-2's own NLL of the true next token. Tertile thresholds come from the train split (easy ≤ 1.63 nats, hard > 4.38 nats) and are applied unchanged to test: easy 13,298, medium 13,535, hard 13,167 examples.
- **Uncertainty:** 95% percentile CIs from 1,000 bootstrap resamples of test articles. These are per layer and group; differences between layers were not tested directly.

## Findings
Test top-1 accuracy, with 95% CIs in brackets:

| layer | all | easy | hard | probe NLL on hard (nats) |
|---|---|---|---|---|
| 0 (embeddings) | 14.9% | 30.0% [28.7, 31.2] | 3.2% [2.8, 3.6] | 9.74 |
| 3 | 20.6% | 42.4% [40.6, 43.8] | 4.0% [3.6, 4.4] | 9.45 |
| 6 | 23.8% | 49.8% [48.2, 51.2] | 4.3% [3.9, 4.8] | 9.12 |
| 9 | 28.5% | 61.5% [60.0, 62.8] | 4.3% [3.8, 4.8] | 8.73 |
| 12 | 31.6% [31.0, 32.2] | 70.4% [69.2, 71.5] | 3.9% [3.5, 4.3] | 8.39 |
| *GPT-2's own top-1* | *35.4%* | *92.1%* | *0.0% (0 of 13,167)* | |
| *always predict " the"* | *5.1%* | *10.2%* | *0.3%* | |

Every layer, the medium group and all CIs are in `results/layerwise_accuracy.csv`.

1. **Across layers:** probe accuracy rises at every layer, from 14.9% to 31.6%, with no intermediate peak.
2. **Easy vs hard:**
   - The gain is concentrated in easy positions (30.0% → 70.4%, still rising at layer 12).
   - Hard positions stay between 3% and 4.5% at every layer. A small rise from layer 0 to layer 5 lies outside the CIs; the dip from layer 11 (4.3%) to layer 12 (3.9%) does not.
   - Probe cross-entropy on hard tokens falls at every layer: the true token gains probability without becoming the top guess.
3. **GPT-2's own accuracy on hard positions:** GPT-2 got 0 of the 13,167 hard positions right. This is an observation, not a mathematical consequence of the threshold. On 55 hard positions GPT-2's top token itself had probability below 1.25%, so a correct top-1 was possible there, but none occurred.

**Decodable vs used.** These probes measure how easily the next token can be read out of each layer with a linear map. They do not show that GPT-2 uses that information at that layer, or uses it in the same way.

## Audit
After the run, I re-checked the code and outputs and rebuilt the exact test positions on CPU, without training probes. The rebuild reproduced the group sizes, the frequency statistics and GPT-2's per-group top-1 counts. The one exception was a single medium-group example (1,913 correct on CPU vs 1,914 on GPU).

The following hold:
- t → t+1 alignment.
- Train-only fitting of the probes, the standardisation and the difficulty thresholds.
- Validation used only for model selection.
- The same test set at every layer.
- No test example shares its 32-token context and target with a sampled training example.

**Parsing flaw.** The notebook starts a new article at every line that looks like a top-level heading.
- 30 lines inside tables and formulas (e.g. `= Goals ; A =`) also match that pattern, so 7 real articles were cut into pieces. For 5 of them the pieces landed in different splits.
- 101 test examples (38 easy, 31 medium, 32 hard) come from a piece whose article also has a piece in train.
- Those pieces share no 33-token span with training examples. Removing them could change any reported accuracy by at most about 0.3 percentage points, so the experiment was not rerun.
- The fix is to also require a blank line before a title.
- One more fragment may exist: the stricter rule finds 601 train titles against about 600 documented articles. This is unresolved.

## Limitations
- **Difficulty is not separated from token frequency.**
  - Hard targets are much rarer among the probe's training labels: median count 14 vs 958 for easy targets (taken per test example), and 9.3% vs 3.7% never seen.
  - The groups do overlap in frequency (1,462 hard examples have targets seen at least 1,000 times), so a frequency-matched comparison is possible in principle. It was not done, because the per-example probe predictions were not kept after the Colab session.
  - The easy/hard gap may be partly or mostly a frequency effect.
- **Probe tuning was simple.**
  - One learning rate (the smallest in the grid) was used for all layers. Layers 2 and 3 were still improving at the 10-epoch cap; layer 0's best epoch was 7, and the rest peaked at epoch 9. Absolute accuracies are therefore probably a little low.
  - On layer 6, the two best learning rates differed by 0.05 nats of validation loss. That is less than every gap between adjacent layers (0.12–0.27 nats), so the overall trend is unlikely to come from tuning.
  - Differences of about one point (within the hard group, or between layers 11 and 12) are not robust.
- **Layer 12 is a different kind of vector:** it is the post-LayerNorm state, while layers 0–11 are the raw residual stream.
- **Probe vs model:** the probe beats GPT-2's top-1 on the medium (20.4% vs 14.1%) and hard (3.9% vs 0%) groups. This is probably a selection effect, since both groups are chosen for GPT-2 giving the true token low probability. This is untested.
- **Scope:** one model, one dataset, one probe seed, contexts of 32–254 tokens.
