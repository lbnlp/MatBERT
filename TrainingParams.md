## Some parameters that could be used on various clusters

### Cori GPU nodes

There are 8 V100 cards, 16GB memory each.

For base model:

- Use fp16 training.
- Maximal batch size per card is 16.
- Average training speed is 1-1.5 batch/s.
 
### Bridges GPU K80

There are 4 K80 cards, 12GB memory each.

For base model:

- Use fp16 training.
- Maximal batch size per card is 8.
- Average training speed is 0.2-0.3 batch/s.
