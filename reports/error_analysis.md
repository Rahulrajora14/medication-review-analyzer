# 🔬 Error analysis: TF-IDF + Logistic Regression

Test reviews: **16,144** | errors: **4,047** (25.1%)

## 1. Most common mistakes

| Mistake | Count | Share of errors |
|---|---:|---:|
| true positive -> predicted neutral | 1,383 | 34.2% |
| true positive -> predicted negative | 869 | 21.5% |
| true negative -> predicted neutral | 641 | 15.8% |
| true neutral -> predicted negative | 420 | 10.4% |
| true neutral -> predicted positive | 384 | 9.5% |
| true negative -> predicted positive | 350 | 8.6% |

## 2. Borderline ratings

**38%** of errors come from ratings 4–7, although only 17% of test reviews have those ratings. A 6/10 and a 7/10 often read almost the same, so part of this error is in the labels themselves.

## 3. Mixed reviews ("worked great, BUT ...")

Accuracy on reviews with contrast words (but / however / although ...): **70.7%** vs **80.4%** on reviews without them.

## 4. Accuracy by review length (words)

| Length | Accuracy |
|---|---:|
| 1-25 | 76.1% |
| 26-75 | 76.0% |
| 76-150 | 74.0% |
| 150+ | 74.2% |

## 5. Most confident mistakes

| Rating | True | Predicted | Confidence | Review |
|---:|---|---|---:|---|
| 6 | neutral | negative | 99% | Omg I was given Stadol in my arm where my IV was to start. I went to ER for throwing up and throat burning and right side of my throat was closing up like swollen and on fire. They took blood, a pee test and came back with zofran in my IV and had me drink ... |
| 5 | neutral | positive | 99% | The best sleep and no side effects |
| 6 | neutral | negative | 97% | Does not work |
| 8 | positive | negative | 96% | I started this birth control pill 12 days ago. I'm 18 years old and take it for bad cramps and excessive bleeding. Ever since I started it I've had extremely debilitating nausea, anxiety, and loss of appetite. I stopped taking it yesterday in hopes of the ... |
| 6 | neutral | negative | 96% | I took this for 4 months. It gave me awful almost dangerous constipation . I felt dumb most of the time. I was never sad nor happy. I wouldn't recommend this unless nothing else worked. |
| 8 | positive | negative | 95% | A tried and true fever reducer, why waste your money on anything shiny |

## 6. Sample neutral reviews the model missed

| Rating | Predicted | Review |
|---:|---|---|
| 6 | positive | I am 43 years old I was recently prescribed gabapentin for morphine withdrawal maintenance. I'm kinda happy with gabbys it really calms me down and helps with nerve and bone pains, I really only have one major side effect from gabapentin which is if I stand ... |
| 5 | positive | I was on Abilify for a year and a half. In that time I gained 30lbs. It also made me very forgetful. I felt like there was a fog in my brain that just wouldn't go away. The med did work but I can't stand feeling like I'm starving every second of the day. I ... |
| 5 | negative | I started this drug about 4 weeks ago. Since I started it I have been vomiting and having diarrhea constantly. I am also more agitated and annoyed than usual, and I have been manic several times (I'm actually taking it for bipolar II, not MDD). It doesn't ... |
| 5 | negative | Had severe sinus problems tried over the counter medicines didn't work diagnosed with bacterial acute sinusitis. Was given medication of levoflaxacin 500 mg . Am feeling better but lots of side effects. Stomach upset depression anxiety. Please do not take ... |
