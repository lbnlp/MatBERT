## Some parameters that could be used on various clusters

### Cori GPU nodes

There are 8 V100 cards, 16GB memory each.

For base model:

- Use fp16 training.
- Maximal batch size per card is 16.
- Average training speed is 1-1.5 batch/s.
 