# Repository for Image Bias Audit Study

This repository contains data, scripts, and supporting materials for a study investigating demographic and stylistic patterns in AI-generated images.  
The dataset includes 880 images generated from multiple models using role-based prompts, along with both human-coded and automated annotations.

**NOTE: Please download the entire repository to be able to view the "generated images"**


## Repository Structure

###  Data & Results
- **Generated Images (880 Images)**  
  Complete set of generated images across roles and models.

- **Images Data (Analysis per Model)**  
  Excel sheet with demographic and stylistic attributes aggregated by model.

- **Images Data (Analysis per Role)**  
  Excel sheet with results aggregated by role prompts.

- **Images Data**  
  Master Excel dataset containing all coded attributes (e.g., gender, race/ethnicity, age group, facial expression, composition).

- **Additional Analysis per Models**  
  Supplementary breakdowns and figures for model-level comparisons.

###  Analysis Scripts
- **analysis_notebook.py**  
  Python script for reproducing analyses and generating visualizations.

- **DeepFace Analysis and Script**  
  Scripts and outputs from the DeepFace library, used for automated demographic and sentiment analysis.

###  Documentation
- **Coding Book (Used for coder training)**  
  Codebook guiding human coders in classifying demographic and stylistic attributes.

- **Image Analysis Prompt (Used to Prompt Models)**  
  List of text prompts used to generate the dataset across different roles and models.

- **Image Replications Table**  
  Image Generation Attempts and Regenerations by Role and Model



## Usage

1. **Explore the data**: Start with `Images Data.xlsx` for the full annotated dataset.  
2. **Reproduce analyses**: Use `analysis_notebook.py` or scripts in `DeepFace Analysis and Script/`.  
3. **Understand methodology**: Refer to the `Coding Book` and `Image Analysis Prompt`.  
4. **Check extended results**: See `Additional Analysis per Models/` for supplementary findings.

---

## Citation
If you use this repository, please cite the associated **forthcoming publication** (details to be added after review).

---

## License
This repository is released for academic and non-commercial use only.
