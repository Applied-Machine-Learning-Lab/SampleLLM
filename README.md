This is the repository of the **SampleLLM** framework in the submission "**Optimizing Tabular Data Synthesis in Recommendations**"

- Note that the repository currently provides only the stripped-down primary source code and examples for code reference, and **does not guarantee that the project will run directly**.

- Due to GitHub file size limitations, we only provide **sample datasets** as references here.

- The operation process of the project is as follows:
  - Use process_data.py to divide the data set into training, validation, and test sets.
  - Use data_generation.py for the first stage of SampleLLM to generate sufficient synthetic data in text form.
  - Use text_process.py in the corresponding dataset folder to summarize and convert multiple synthetic data into tabular form.
  - Perform the second stage of SampleLLM using samply.py to filter the synthetic data.
  - Use augmentation.py or mle.py to calculate the augmentation utility or MLE utility metrics.
