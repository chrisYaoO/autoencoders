from methods import *

with open('config_ae.json') as config_file:
    config = json.load(config_file)

dataset = config['dataset']
model_name = config['model_name']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
valid_rate = config['valid_rate']
shuffle = config['shuffle']
model_filename = f"{model_name}_batch_{batch_size}_epoch_{num_epochs}_valid_{valid_rate}"
# 我们将阈值设置为等于自动编码器的训练损失


x_test = pd.read_csv('processed/' + dataset + '/x_test.csv')
y_test = pd.read_csv('processed/' + dataset + '/y_test.csv')
autoencoder = load_model('results/' + dataset + '/' + model_filename + '.h5')
history = pd.read_csv('results/' + dataset + '/training_history_' + model_filename + '.csv')

threshold = history["loss"].tolist()[-1]
testing_set_predictions = autoencoder.predict(x_test)
test_losses = calculate_losses(x_test, testing_set_predictions)

testing_set_predictions = np.zeros(len(test_losses))
testing_set_predictions[np.where(test_losses > threshold)] = 1

recall = recall_score(y_test, testing_set_predictions)
precision = precision_score(y_test, testing_set_predictions)
f1 = f1_score(y_test, testing_set_predictions)
print("Performance over the testing data set \n")
print("Accuracy : {} \nRecall : {} \nPrecision : {} \nF1 : {}\n".format(accuracy, recall, precision, f1))
