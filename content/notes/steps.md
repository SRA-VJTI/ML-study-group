---
author: "Kush Kothari"
title: "Sample content"
date: 2022-10-18T11:08:37+05:30
draft: false
tags:
    - Demo
    - Demo2
---

Just copy stuff from this file and use it for reference as needed.

# Heading 1
## Heading 2
### Heading 3
#### Heading 4
##### Heading 5

> Indented stuff.

### Inline styles
**strong**, *emphasis*, ***strong and emphasis***,`code`, ~~strikethrough~~, :joy:🤣, [Link](https://example.com)

### Inline Image
![img](https://picsum.photos/600/400/?random)

### Code block (with python syntax highlighting)
```bash
cd ML-study-group
hugo new notes/filename.md
```

```python
for post in posts:
    print(post)
```

### LaTeX

$$\(\begin{aligned} \dot{x} & = \sigma(y-x) \\ \dot{y} & = \rho x - y - xz \\ \dot{z} & = -\beta z + xy \end{aligned} \)$$

$$ \left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right) $$

$$\[ \frac{1}{\Bigl(\sqrt{\phi \sqrt{5}}-\phi\Bigr) e^{\frac25 \pi}} = 1+\frac{e^{-2\pi}} {1+\frac{e^{-4\pi}} {1+\frac{e^{-6\pi}} {1+\frac{e^{-8\pi}} {1+\ldots} } } } \]$$

### Table

| Left-Aligned  | Center Aligned  | Right Aligned |
| :------------ | :-------------: | ------------: |
| col 3 is      | some wordy text |         $1600 |
| col 2 is      |    centered     |           $12 |
| zebra stripes |    are neat     |            $1 |

### Lists
* Unordered list item 1.
* Unordered list item 2.

1. ordered list item 1.
2. ordered list item 2.
   + sub-unordered list item 1.
   + sub-unordered list item 2.
     + [x] something is DONE.
     + [ ] something is NOT DONE.


### Figures
This theme has 3 CSS classes made for figure elements:

* `big`: images will break the width limit of main content area.
* `left`: images will float to the left.
* `right`: images will float to the right.
  
If a figure has no class set, the image will behave just like a normal markdown image: `![]()`.

Here's some examples, please be aware that these styles only take effect when the page width is over 1300px.

{{< figure src="https://via.placeholder.com/1600x800" alt="image" caption="figure-normal (without any classes)" >}}

{{< figure src="https://via.placeholder.com/1600x800" alt="image" caption="figure-big" class="big" >}}

{{< figure src="https://via.placeholder.com/400x280" alt="image" caption="figure-left" class="left" >}} 
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis vel mollis purus. Sed ut lectus augue. Proin consectetur augue et arcu mollis, ut condimentum est posuere. Nullam vehicula rhoncus lacus, ut pulvinar urna mattis nec. Proin venenatis commodo nisi, hendrerit tincidunt lorem luctus eu. Proin dapibus aliquet ultricies. Integer felis quam, venenatis quis suscipit vel, sodales dictum quam. Sed non ex et dui elementum dictum et quis ante. Vivamus dapibus, leo quis elementum auctor, erat metus faucibus tellus, quis aliquet erat orci vitae lectus.