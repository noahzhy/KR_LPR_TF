# Dataloader

## Dataset

$N$ is the number of tokens in the sentence. $T$ is the time steps of the model. Forcing alignments are working when $T \ge N$.

Force alignments are generated by the following formula:

$$ T = \lceil \frac{N}{2} \rceil + N $$

Average alignments are generated by the following formula:

$$ T = aN, a \ge 2 $$

where $N$ is the number of tokens in the sentence.
