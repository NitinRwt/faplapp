
## 🧮 **Step-by-Step Interpolation Calculation**

Suppose:
- You detected the **positions** of the numbers:  
  - 10 is at **(x₁, y₁)**  
  - 15 is at **(x₂, y₂)**
- The needle tip is at **(xₙ, yₙ)**

The code compares **only the X-coordinates** (horizontal axis) to estimate where the needle lies between the two numbers.

---

### ✅ **Step 1: Compute Distance Along X-axis**

```python
total_x_distance = right_position[0] - left_position[0]  # x₂ - x₁
needle_x_distance = needle_tip[0] - left_position[0]     # xₙ - x₁
```

---

### ✅ **Step 2: Ratio of Needle’s Position Between Two Numbers**

```python
ratio = needle_x_distance / total_x_distance
```

So, if:
- `needle_x_distance = 25`
- `total_x_distance = 50`

Then:
```python
ratio = 25 / 50 = 0.5
```

---

### ✅ **Step 3: Interpolated Value**

```python
interpolated_value = left_value + (ratio * (right_value - left_value))
```

Using our 10 → 15 example:

```python
interpolated_value = 10 + (0.5 * (15 - 10)) = 10 + 2.5 = 12.5
```

Then it rounds this to **one decimal place** and returns:
```python
12.5
```

---

### 🔍 Why Use X-Axis Only?

Analog dials are usually laid out in a **semi-circular or horizontal fashion**, so checking the **needle's horizontal position** (X-axis) gives a good estimation of its value between numbers.
