# ProteinPicker

This Github repository holds all of the code used to train the atlas and tomogram models featured in our CS562 final Project

As a general overview of how the repository is structured:

- `archive`: holds various code files that are no longer used by the ProteinPicker project
- `imgs`: holds 2D slices of MRC files, back from when the model was originally a 2D CNN model. These images are no longer used for the updated 3D CNN model.
- `mrc`: holds all of the mrc files used for the 3D CNN model
    - `atlas-mrc` holds all of the atlas MRC files used for training the atlas 3D CNN model
    - `tomogram-mrc` holds all of the tomogram MRC files used for training the tomogram 3D CNN model
- `saved_model`: stores all models featured in the ProteinPicker project. For the purposes of the final project, there are only two models that were featured in the paper.
    - `atlas_3d`
        - `atlas_3d_empty_model.keras`: "atlas" model featured in the paper
        - `Atlas3DModel.ipynb`: Jupyter notebook used to create the aforementioned model
    - `tomogram_3d`:
        - `tomogram_3d_empty_model.keras`: "tomogram" model featured in the paper
        - `Tomogram3DModel.ipynb`: Jupyter notebook used to create the aforementioned model
    - The remaning folders were unused in the final paper, and serve as "milestones" for documenting the journey for improving our CNN models.
