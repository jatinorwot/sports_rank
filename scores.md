# Sports Action Photo Scoring System
## Complete Score Calculation Guide

### Table of Contents
1. [Overview](#overview)
2. [Individual Score Components](#individual-score-components)
3. [Composite Scores](#composite-scores)
4. [Final Score Calculation](#final-score-calculation)
5. [Score Interpretation](#score-interpretation)
6. [Examples](#examples)

---

## Overview

The Sports Action Photo Ranking System uses a multi-metric approach to evaluate photos, combining 10+ different scores to identify peak action moments. Each metric is normalized to a 0-10 scale for consistent comparison.

### Score Categories
- **Pose-Based Scores**: Derived from body keypoint analysis
- **Interaction Scores**: Ball-player proximity and context
- **Quality Scores**: Technical aspects like sharpness and exposure
- **Composite Scores**: Combinations of multiple metrics

---

## Individual Score Components

### 1. Pose Confidence Score (0-10)
**Purpose**: Measures how reliably the pose detection system identified the person

**Calculation**:
```python
pose_confidence = mean(visibility of all 33 landmarks)
```

**Details**:
- MediaPipe provides visibility (0-1) for each landmark
- Average visibility across all 33 body keypoints
- Higher score = clearer, more visible person

**Example**:
- All landmarks visible (1.0 each) → Score: 10.0
- Half landmarks visible (0.5 average) → Score: 5.0

---

### 2. Body Extension Score (0-10)
**Purpose**: Measures how spread out/extended the body is (indicates dynamic movement)

**Calculation**:
```python
# 3D spread of all body points
positions = [x, y, z for all landmarks]
spread_3d = standard_deviation(positions)
total_spread = magnitude(spread_3d)

# Individual limb measurements
limb_lengths = [
    distance(right_shoulder → right_wrist),
    distance(left_shoulder → left_wrist),
    distance(right_hip → right_ankle),
    distance(left_hip → left_ankle)
]
max_extension = max(limb_lengths)

body_extension = min((total_spread × 5) + (max_extension × 5), 10.0)
```

**Scoring Logic**:
- Compact pose (standing still) → Low score (~2-3)
- Wide stance or reach → Medium score (~5-6)
- Full extension (jumping/lunging) → High score (~8-10)

---

### 3. Athletic Pose Score (0-10)
**Purpose**: Identifies athletic stances and positions common in sports

**Calculation Components**:

#### a) Knee Bend Score (0-3 points)
```python
knee_angle = angle(hip → knee → ankle)
if 120° < knee_angle < 170°:
    knee_bend_score = 3.0  # Athletic stance
else:
    knee_bend_score = 0.0  # Standing straight
```

#### b) Stance Width Score (0-3 points)
```python
stance_width = |left_ankle.x - right_ankle.x|
stance_score = min(stance_width × 10, 3.0)
```

#### c) Forward Lean Score (0-2 points)
```python
torso_angle = |nose.y - hip_center.y|
lean_score = min(torso_angle × 5, 2.0)
```

#### d) Dynamic Position Score
```python
shoulder_tilt = |left_shoulder.y - right_shoulder.y|
hip_tilt = |left_hip.y - right_hip.y|
dynamic_score = (shoulder_tilt + hip_tilt) × 10
```

**Final Athletic Score**:
```python
athletic_pose = min(knee_bend + stance + lean + dynamic, 10.0)
```

---

### 4. Motion Intensity Score (0-10)
**Purpose**: Measures indicators of fast/dynamic movement

**Calculation Components**:

#### a) Wrist Height Score (0-4 points)
```python
shoulder_height = (left_shoulder.y + right_shoulder.y) / 2
if left_wrist.y < shoulder_height OR right_wrist.y < shoulder_height:
    wrist_height_score = 4.0  # Arms raised (hitting motion)
else:
    wrist_height_score = 0.0
```

#### b) Arm Extension Score (0-3 points)
```python
arm_angle = angle(shoulder → elbow → wrist)
if arm_angle > 150°:
    extension_score = 3.0  # Fully extended arm
else:
    extension_score = 0.0
```

#### c) Body Rotation Score
```python
shoulder_line = |left_shoulder.x - right_shoulder.x|
hip_line = |left_hip.x - right_hip.x|
rotation_score = |shoulder_line - hip_line| × 10
```

#### d) Feet Movement Score
```python
feet_spread = |left_ankle.x - right_ankle.x|
feet_elevation = max(0, hip.y - min(left_ankle.y, right_ankle.y))

movement_score = (feet_spread × 5) + (feet_elevation × 10)
```

**Final Motion Score**:
```python
motion_intensity = min(sum_of_all_components, 10.0)
```

---

### 5. Ball Interaction Score (0-10)
**Purpose**: Measures how close the ball is to the player's hitting zone

**Calculation Components**:

#### a) Proximity Score (0-8 points)
```python
# Calculate distances to hands and extended racket positions
distances = [
    distance(left_wrist → ball),
    distance(right_wrist → ball),
    distance(left_wrist + 0.5×(wrist-elbow) → ball),  # Racket estimate
    distance(right_wrist + 0.5×(wrist-elbow) → ball)
]
min_distance = min(distances)

proximity_score = max(0, (1 - min_distance) × 8)
```

#### b) Height Bonus (0-2 points)
```python
ball_height_ratio = 1 - (ball_y_position / image_height)
height_bonus = ball_height_ratio × 2
```

#### c) Motion Blur Bonus (0-1 point)
```python
ball_region = image[ball_bbox]
laplacian_variance = variance(laplacian_filter(ball_region))
blur_score = 10 - (laplacian_variance / 10)  # Higher blur = faster ball
motion_bonus = min(blur_score × 0.5, 1.0)
```

#### d) Action Context Bonus (0-2 points)
```python
if action_type in ['forehand', 'backhand', 'serve'] AND min_distance < 0.3:
    action_bonus = 2.0
elif action_type == 'volley' AND ball_y < 0.5:
    action_bonus = 1.5
else:
    action_bonus = 0.0
```

**Final Ball Interaction Score**:
```python
ball_interaction = min(proximity + height + motion + action + equipment_bonus, 10.0)
```

---

### 6. Sharpness Scores (0-10)

#### a) Overall Sharpness
```python
gray_image = convert_to_grayscale(full_image)
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
variance = laplacian.var()
overall_sharpness = min(variance / 100, 10.0)
```

#### b) Subject Sharpness
```python
# Focus on center 50% of image where subject likely is
center_region = image[h/4:3h/4, w/4:3w/4]
center_laplacian = cv2.Laplacian(center_region, cv2.CV_64F)
subject_variance = center_laplacian.var()
subject_sharpness = min(subject_variance / 100, 10.0)
```

**Interpretation**:
- Variance < 100: Very blurry (Score: 0-1)
- Variance 100-500: Slightly blurry (Score: 1-5)
- Variance 500-1000: Sharp (Score: 5-10)
- Variance > 1000: Very sharp (Score: 10)

---

### 7. Composition Score (0-10)
**Purpose**: Evaluates photo composition quality

**Sub-components**:

#### a) Framing Score (0-10)
```python
# Check if player is well-positioned in frame
margin = 0.05
if player_bbox touches edges:
    framing_score = 3.0  # Cut off
elif 0.15 < player_area < 0.6:
    framing_score = 9.0  # Good size
else:
    framing_score = 6.0  # Too small/large
```

#### b) Rule of Thirds Score (0-10)
```python
# Check if subject center aligns with power points
thirds_x = [1/3, 2/3]
thirds_y = [1/3, 2/3]
min_distance = min(distances_to_power_points)

if min_distance < 0.1:
    rule_of_thirds = 9.0  # Perfect alignment
elif min_distance < 0.15:
    rule_of_thirds = 7.0  # Good alignment
else:
    rule_of_thirds = 5.0  # Poor alignment
```

#### c) Action Space Score (0-10)
```python
# Space in direction of movement/action
if player_facing_left:
    action_space_available = left_margin
else:
    action_space_available = right_margin

if action_space_available > 0.3:
    action_space = 9.0  # Plenty of space
elif action_space_available > 0.15:
    action_space = 6.0  # Adequate space
else:
    action_space = 3.0  # Cramped
```

**Final Composition Score**:
```python
composition_score = mean([framing, rule_of_thirds, diagonal, negative_space, action_space])
```

---

### 8. Technical Quality Score (0-10)

#### a) Exposure Score (0-10)
```python
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
mean_luminance = mean(L_channel)

if 40 < mean_luminance < 200:
    exposure_score = 9.0  # Well exposed
elif 20 < mean_luminance < 230:
    exposure_score = 6.0  # Acceptable
else:
    exposure_score = 3.0  # Under/over exposed
```

#### b) Contrast Score (0-10)
```python
contrast = standard_deviation(L_channel)
contrast_score = min(contrast / 5, 10.0)
```

**Final Technical Score**:
```python
technical_quality = mean([subject_sharpness, exposure, contrast])
```

---

### 9. Peak Action Score (0-10)
**Purpose**: Composite score for identifying peak action moments

**Calculation**:
```python
peak_action = (
    athletic_pose × 0.2 +
    motion_intensity × 0.3 +
    velocity_indicator × 0.2 +
    symmetry_score × 0.1 +
    action_confidence × 0.2
)

# Bonus for high-value actions
if action_type in ['serve', 'forehand', 'backhand']:
    peak_action += 1.0

peak_action = min(peak_action, 10.0)
```

---

### 10. Additional Metrics

#### Symmetry Score (0-10)
```python
# Asymmetry indicates dynamic movement
asymmetry = 0
for left_point, right_point in paired_landmarks:
    diff_x = |left.x - (1 - right.x)|  # Mirror comparison
    diff_y = |left.y - right.y|
    asymmetry += (diff_x + diff_y)

symmetry_score = min(asymmetry × 2, 10.0)
```

#### Velocity Indicator (0-10)
```python
# Estimated from pose blur and extreme positions
visibility_variance = variance([landmark.visibility for all landmarks])
extreme_positions = count(wrists_very_high OR ankles_very_wide)

velocity_score = (visibility_variance × 20) + (extreme_positions × 2)
velocity_indicator = min(velocity_score, 10.0)
```

#### Player Orientation (0-10)
```python
shoulder_visibility = (left_shoulder.visibility + right_shoulder.visibility) / 2
face_visibility = nose.visibility

orientation_score = (shoulder_visibility × 7) + (face_visibility × 3)
```

---

## Composite Scores

### Action Classification
The system classifies detected poses into specific actions:

```python
def classify_action(landmarks):
    # Extract features
    right_wrist_high = right_wrist.y < right_shoulder.y
    left_wrist_high = left_wrist.y < left_shoulder.y
    both_arms_up = right_wrist_high AND left_wrist_high
    wide_stance = |left_ankle.x - right_ankle.x| > 0.3
    
    # Classification rules
    if both_arms_up AND wrist.y < 0.3:
        return 'serve', confidence=0.8
    elif right_wrist_high AND NOT left_wrist_high:
        return 'forehand', confidence=0.7
    elif left_wrist_high AND NOT right_wrist_high:
        return 'backhand', confidence=0.7
    elif wide_stance AND ankle.y > 0.8:
        return 'lunge', confidence=0.6
    elif wide_stance:
        return 'ready_position', confidence=0.5
    else:
        return 'general_movement', confidence=0.3
```

---

## Final Score Calculation

### Step 1: Apply Base Weights
```python
base_weights = {
    'peak_action_score': 0.25,    # 25% weight
    'ball_interaction': 0.20,      # 20% weight
    'motion_intensity': 0.15,      # 15% weight
    'athletic_pose': 0.10,         # 10% weight
    'subject_sharpness': 0.10,     # 10% weight
    'composition_score': 0.10,     # 10% weight
    'technical_quality': 0.10      # 10% weight
}

weighted_sum = Σ(score[metric] × base_weights[metric])
```

### Step 2: Apply Sport-Specific Modifiers
```python
sport_modifiers = {
    'pickleball': {
        'ball_interaction': 1.2,    # 20% boost
        'athletic_pose': 1.1        # 10% boost
    },
    'tennis': {
        'motion_intensity': 1.2,    # 20% boost
        'peak_action_score': 1.1    # 10% boost
    },
    'badminton': {
        'motion_intensity': 1.3,    # 30% boost
        'ball_interaction': 0.9     # 10% reduction
    }
}

# Apply modifiers
for metric, modifier in sport_modifiers[sport].items():
    weighted_scores[metric] *= modifier
```

### Step 3: Add Bonuses
```python
total_score = weighted_sum

# Bonus conditions
if pose_detected AND ball_detected:
    total_score += 1.0  # Major bonus

if action_type in ['serve', 'forehand', 'backhand']:
    total_score += 0.5  # Action bonus

# Cap at maximum
final_score = min(total_score, 10.0)
```

---

## Score Interpretation

### Score Ranges

| Score Range | Quality Level | Description |
|-------------|--------------|-------------|
| 0.0 - 2.0 | Very Poor | Blurry, no action, major issues |
| 2.0 - 4.0 | Poor | Some issues, minimal action |
| 4.0 - 5.5 | Below Average | Decent quality but not peak action |
| 5.5 - 7.0 | Good | Clear action moment captured |
| 7.0 - 8.5 | Excellent | Peak action with good quality |
| 8.5 - 10.0 | Outstanding | Perfect timing and exceptional quality |

### What Makes a High-Scoring Photo?

**Essential Elements (Must Have)**:
- Clear, sharp subject (sharpness > 6)
- Detected pose (confidence > 0.5)
- Athletic position (athletic_pose > 5)

**High-Value Elements (Big Score Boost)**:
- Ball near contact point (ball_interaction > 7)
- Peak action moment (motion_intensity > 7)
- Specific action detected (serve/forehand/backhand)

**Quality Factors (Refinement)**:
- Good composition (centered, rule of thirds)
- Proper exposure and contrast
- Player facing camera

---

## Examples

### Example 1: Perfect Serve (Score: 9.2)
```
- Peak Action Score: 9.5 (arms up, back arched)
- Ball Interaction: 8.0 (ball at peak height)
- Motion Intensity: 9.0 (full extension)
- Athletic Pose: 8.5 (perfect form)
- Subject Sharpness: 8.0 (crisp detail)
- Composition: 7.5 (well framed)
- Technical Quality: 8.0 (good exposure)

Weighted Total: 8.7 + Bonuses (1.0 + 0.5) = 9.2
```

### Example 2: Average Rally Shot (Score: 5.8)
```
- Peak Action Score: 6.0 (some movement)
- Ball Interaction: 4.0 (ball distant)
- Motion Intensity: 5.0 (moderate movement)
- Athletic Pose: 6.0 (ready position)
- Subject Sharpness: 7.0 (clear)
- Composition: 6.0 (acceptable)
- Technical Quality: 7.0 (well lit)

Weighted Total: 5.8 + No bonuses = 5.8
```

### Example 3: Poor Quality (Score: 2.1)
```
- Peak Action Score: 2.0 (standing still)
- Ball Interaction: 0.0 (no ball detected)
- Motion Intensity: 1.0 (static)
- Athletic Pose: 2.0 (casual stance)
- Subject Sharpness: 3.0 (blurry)
- Composition: 4.0 (off-center)
- Technical Quality: 3.0 (underexposed)

Weighted Total: 2.1 + No bonuses = 2.1
```

---

## Key Insights

1. **Action Detection is Crucial**: Photos with detected actions (serve, forehand, backhand) automatically score higher due to bonuses and context-aware scoring.

2. **Ball Proximity Matters**: The ball_interaction score has high weight (20%) and gets sport-specific boosts, making ball contact moments highly valued.

3. **Technical Quality is Secondary**: While important, sharpness and exposure (10% each) matter less than capturing the right moment.

4. **Sport-Specific Optimization**: The modifier system ensures each sport's key moments are properly emphasized (e.g., ball interaction for pickleball).

5. **Compound Effects**: Multiple high scores create exponential improvements through bonuses and weighted combinations.

This scoring system successfully identifies and ranks peak action moments while maintaining technical quality standards, ensuring the best sports photos rise to the top of the rankings.
