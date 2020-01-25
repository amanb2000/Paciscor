# Optical Character Recognition

## Input

1. `path`: Path to .jpg file we are analyzing.
2. `coords`: Tuple of tuples. Each sub-tuple contains coordinates of the top left and of the bottom right corner of each **ad block**.

### Example

```python
path = '../../src/flyers/week_1_page_1.jpg'
coords = (
    (
        (188, 23),
        (19, 233)
    ),
    ...
)
```

## Output

```python
`output` = `tuple`[
    `dict`{
        `text`: "$5 Off!",
        `color`: "red", # red, yellow, blue, green, etc.
        `size`: 15, # integer
        `font`: "..." # remains to be seen as I do more research. For now, don't worry too much about this.
        `styling`: "..." # remains to be seen as I do more research.
    },
    `dict`{
        ...
    }
]
```
