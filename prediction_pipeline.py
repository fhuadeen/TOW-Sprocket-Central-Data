def prediction_pipeline(data, processor, transformer, predictor):
    """ Pipeline to run all transformers and make prediction on new
    Sprocket Central data"""

    # prepare data for further transformation
    tidied_data = processor.transform(data)

    # transform categorical and numerical variables on missing values,
    # standardisation (numerical) and OneHotEncoding (categorical)
    transformed_data = transformer.transform(tidied_data)
    predictions = predictor.predict(transformed_data)

    return predictions
