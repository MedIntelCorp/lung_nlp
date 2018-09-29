## NLP Nodule Learner

### Description
This repository contains a supervised classification model for predicting whether a patient has a pulmonary nodule based on the doctor's comments in a report about the patient in question. For patients predicted to have a pulmonary nodule, the size and location of that nodule will also be extracted.

### Usage

To use, first create an instance of the `NoduleLearner` class, then call the `transform_predict` instance method on a file containing the doctor's report from which to make the prediction, as follows:

```python
nl = NoduleLearner()
prediction = nl.transform_predict(file_name)
```

The `transform_predict()` method by default doesn't actually make predictions, but rather transforms the data and caches it, later making the actual predictions once the `cache_size` is reached (default size is 500, but this can be set when instantiating the `NoduleLearner` class). If the learner finishes looping through the files and there are still some left to predict, simply call `nl.dump_predictions()` (you'll stil need to set the `probability` and `vetting` arguments) to make the predictions on the rest of the records left in the cache.

There are three arguments to `transform_predict()`, the latter two of which default to `False`:

* `file` (`str`): the file on which to perform the predictions
* `probability` (`bool`): whether or not to output predictions for the nodule classifications
* `vetting` (`bool`): whether to output data needed to vet the extractions (more on that below)

When not vetting the extractions (`vetting=False`), the model outputs the following data for each record:

* `directory (str)`: the folder from which the file came (can be useful for traceability)
* `filename (str)`: the filename of the file, minus the extension
* `max_nodule_change (float)`: if there is a previous record for the patient which also contains a nodule size, this is the difference in size of the largest nodule seen in the patient record
* `max_nodule_location (str)`: the location of the largest nodule extracted; one of left upper lobe, lingula, left lower lobe, right upper lobe, right middle lobe, right lower lobe, or '' (indicating no location was found)
* `max_nodule_lung (str)`: the lung in which the largest extracted nodule was located; one of left, right, or ''
* `max_nodule_size (str)`: the size of the largest nodule, in millimeters
* `evidence (str)`: the phrases extracted that contained terms indicating the possible presence of nodules, from which the largest nodule was extracted, if there was one
* `pid (str)`: the patient ID for the record in question
* `prediction (int)`: the prediction for whether or not the report indicates the presence of a pulmonary nodule; either 0 (negative) or 1 (positive)
* `probability (float)`: (if `probability=True`) the model's confidence in the prediction, on 0 (least confident) to 1 (most confident) scale
* `report_date`: the date of the report in question (used for tracking the growth of the largest nodule)
* `prev_max_date`: the date of the previous largest nodule (used for tracking the growth of the largest nodule)

If `vetting` is set to `True`, the following fields will also be present in the output:

* `is_nodule (int)`: whether or not the phrase in question contains evidence of a pulmonary nodule, 0 for negative, 1 for positive
* `nodule_location (str)`: the extracted location of the nodule in the phrase in question (same parameters as for `max_nodule_location` above)
* `nodule_lung (str)`: the extracted lung of the nodule in the phrase in question
* `nodule_size (str)`: the extracted size of the nodule, in string form (i.e. "5 mm", etc.)
* `nodule_size_numeric (float)`: the extracted size of the nodule, in float form (same as for `max_nodule_size` above)
* `phrase_counter (int)`: an integer value to differentiate the phrases pulled out of a record that contain a term for a nodule

Each of the values above, minus the `phrase_counter`, must be vetted in order to measure and improve the performance of the nodule extractor (the part that identifies size, location, and lung of the largest nodule).

The output will be in the form of a JSON array.