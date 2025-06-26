## ZERO SHOT NON COT PROMPT
non_cot_text_instruction = """
### **Task**:
Analyze the provided text and determine if it's sarcastic.

---

### **Output Format**  

```
**Label**:  (SARCASTIC / NOT SARCASTIC)
**Reasoning**: (Your brief reasoning)   
```

---

### **Important Notes**:  
- Provide outputs **exactly** in the specified format.
- Do **not** include any additional commentary or explanation outside the specified format.
- The possible values of label are "SARCASTIC" or "NOT SARCASTIC"
"""

non_cot_image_instruction = """
### **Task**:
Analyze the provided image and determine if it's sarcastic.

---

### **Output Format**  

```
**Label**:  (SARCASTIC / NOT SARCASTIC)
**Reasoning**: (Your brief reasoning)  
```

---

### **Important Notes**:  
- Provide outputs **exactly** in the specified format.
- Do **not** include any additional commentary or explanation outside the specified format.
- The possible values of label are "SARCASTIC" or "NOT SARCASTIC"
"""

non_cot_mm_instruction = """
### **Task**:
Analyze the provided text and image and determine if it's sarcastic.

---

### **Output Format**  

```
**Label**:  (SARCASTIC / NOT SARCASTIC)
**Reasoning**: (Your brief reasoning)   
```

---

### **Important Notes**:  
- Provide outputs **exactly** in the specified format.
- Do **not** include any additional commentary or explanation outside the specified format.
- The text may come in two distinct forms: text overlaid on image, and text provided separately. Look out for both.
- The possible values of label are "SARCASTIC" or "NOT SARCASTIC"
"""

## ZERO SHOT COT PROMPTS
cot_text_instruction = """
### **Premise**:
You are a sarcasm analysis agent tasked with analyzing a given text.

---

### **Task**:
This task involves analyzing the linguistic content of the text to determine if it contains sarcasm. 
You need to determine if the text is meant to be sarcastic or not.

---

### **Recommendations**:
1. Apply the principle of charity: Before labeling content as sarcastic, first attempt to interpret text at face value. Many statements are genuinely positive or negative without ironic intent. Only identify sarcasm when there is clear, unambiguous evidence of intentional contradiction.
2. Distinguish between humor types: Not all humor is sarcasm. Differentiate between genuine jokes, hyperbole, exaggeration for emphasis, and actual sarcasm. Sarcasm specifically requires saying the opposite of what is meant with the intent to mock or critique.

### **Output Format**  

```
**Thinking**:  (Elaborate analysis of the text and possible cues of sarcasm)   
**Label**:  (SARCASTIC / NOT SARCASTIC)  
**Reasoning**:  (Concise explanation)  
```
---

### **Important Notes**:  
- Provide outputs **exactly** in the specified format.
- Do **not** include any additional commentary or explanation outside the specified format.
- The possible values of label are "SARCASTIC" or "NOT SARCASTIC".
- Keep your **Reasoning** concise and directly related to your analysis.
"""

cot_image_instruction = """
### **Premise**:
You are a visual sarcasm analysis agent tasked with analyzing a given image.

---

### **Task**:
This task involves analyzing the visual content of the image to determine if it conveys sarcasm. 
You need to examine visual cues, composition, and any text overlaid on the image to determine if it is meant to be sarcastic or not.

---

### **Recommendations**:
1. Apply the principle of charity: Before labeling content as sarcastic, first attempt to interpret the image at face value. Many images are genuinely positive or negative without ironic intent. Only identify sarcasm when there is clear visual evidence of intentional contradiction or irony.
2. Distinguish between humor types: Not all visual humor is sarcasm. Differentiate between genuine visual jokes, hyperbole, exaggeration for emphasis, and actual sarcasm. Visual sarcasm specifically requires showing something that contradicts what is actually meant, often to mock or critique.

### **Output Format**  

```
**Thinking**:  (Elaborate analysis of the visual elements and possible cues of sarcasm)   
**Label**:  (SARCASTIC / NOT SARCASTIC)  
**Reasoning**:  (Concise explanation)  
```
---

### **Important Notes**:  
- Provide outputs **exactly** in the specified format.
- Do **not** include any additional commentary or explanation outside the specified format.
- The possible values of label are "SARCASTIC" or "NOT SARCASTIC".
- Keep your **Reasoning** concise and directly related to your analysis.
"""

cot_mm_instruction = """
### **Premise**:
You are a multimodal sarcasm analysis agent tasked with analyzing the given text and image where a social media user comments on a social media post.

---

### **Task**:
This task involves analyzing and explaining information from two distinct modalities: the linguistic content of the text(s) and the visual cues present in the image. 
You need to determine evidence of sarcasm in each modality and integrate all these insights to determine the overall intent of the user. 
To be specific, you need to determine if the social media post is meant to be sarcastic or not.

---

### **Recommendations**:
1. Apply the principle of charity: Before labeling content as sarcastic, first attempt to interpret text and images at face value. Many social media posts are genuinely positive or negative without ironic intent. Only identify sarcasm when there is clear, unambiguous evidence of intentional contradiction or mismatch between modalities.
2. Distinguish between humor types: Not all humor is sarcasm. Differentiate between genuine jokes, hyperbole, exaggeration for emphasis, and actual sarcasm. Sarcasm specifically requires saying the opposite of what is meant with the intent to mock or critique, while many humorous posts lack this subversive quality.

### **Output Format**  

```
**Thinking**:  (Elaborate analysis of each modalities, the integration of insights, and possbile cues of sarcasm)   
**Label**:  (SARCASTIC / NOT SARCASTIC)  
**Reasoning**:  (Concise explanation)  
```
---

### **Important Notes**:  
- Provide outputs **exactly** in the specified format.
- Do **not** include any additional commentary or explanation outside the specified format.
- The text may come in two distinct forms: text overlaid on image, and text provided separately. Look out for both.
- The possible values of label are "SARCASTIC" or "NOT SARCASTIC".
- Keep your **Reasoning** concise and directly related to your analysis.
"""