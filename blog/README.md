# Blog figures

The headline figure for the blog post: what offline self-distillation does, in one loop.

- `blog/fig1.html`: the animation (self-contained, open in a browser; loops, about 45 s). Reduced motion
  shows a still of the key moment.
- `blog/storyboard.html`: the same figure as static frames with notes, 1 → 4.9.
- `blog/distributions.html`: the last round of options for drawing the two distributions (exploration,
  not used directly; the figure uses G4a).

The data in the figure is illustrative, not from our runs: a made-up turn ("Let me also run the tests.")
with distributions shaped like the real ones. Real rollouts (`runs/train250/trajectories.jsonl`) were
only used for the look and feel.

## What the figure shows, in order

1. The model's rollout on a task: a fixed sequence of turns.
2. The same rollout with a hint written into the middle ("You're taking too long. Wrap up."). The turns
   after it are the same tokens, shifted over.
3. One model reads both: the student reads the plain rollout, the teacher (same model) reads it with
   the hint. Only the turns after the hint are supervised.
4. The target tokens zoom out of one fixed turn ("…" for the rest). For each token: the student's
   context slides in and its distribution grows below the token, then the teacher's context (with the
   hint) and its distribution above. Compare (KL): green when they agree, red where the hint changes
   what the model would do ("also": the teacher would rather submit or wrap up). Each token's loss
   drops into a strip at the bottom.
5. A 3-second speed-up through every token of every fixed turn, the losses piling up on the left; then
   the strip moves to the center: the training signal, one loss per target token.

## Takeaways from making it

1. **Realism distracts.** Terminal panes, typing and real commands made it look like a product demo.
   Plain blocks for turns, cells for tokens and made-up distributions explain the method better.
2. **The idea to make visible is "same fixed tokens, two contexts".** What made it land: inserting
   the hint into an intact copy of the rollout (rewinding read as regenerating), outlining the
   supervised turns, zooming the targets out of the rollout with a "…", and laying each model's
   context right before the token it scores.
3. **Small encoding choices change the reading.** Keep the focus on one token at a time (persistent
   colors suggested one token's score affects the next); draw distributions so they look like
   distributions (teacher above, student below, √p heights, a smooth curve behind, named tokens);
   end on the per-token training signal, not on a single mean loss or a claimed payoff.
