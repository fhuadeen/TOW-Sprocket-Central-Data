from complete_transformer import TidyData
from complete_transformer import pipeline_transformer
from sklearn.linear_model import ElasticNet


def model_pipeline(data_attributes, train_data, num_features, cat_features, y):

    data_tidying = TidyData(data_attributes)
    processed_data = data_tidying.fit_transform(train_data)

    data_transformer = pipeline_transformer(num_features, cat_features)
    transformed_data = data_transformer.fit_transform(processed_data)

    model = ElasticNet(alpha=10.0, l1_ratio=0.7, max_iter=2000)
    complete_model = model.fit(transformed_data, y)

    return complete_model
