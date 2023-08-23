from methods import *
from model import *
from tensorflow.python.keras.optimizer_v2 import adam

with open('config_ae.json') as config_file:
    config = json.load(config_file)

dataset = config['dataset']
model_name = config['model_name']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
valid_rate = config['valid_rate']
shuffle = config['shuffle']
model_filename = f"{model_name}_batch_{batch_size}_epoch_{num_epochs}_valid_{valid_rate}"

if dataset == 'wadi':
    x_train = pd.read_csv('processed/wadi/x_train.csv')
    x_train = x_train.iloc[1:, :]
    print(x_train.shape)
else:
    x_train = pd.read_csv('processed/kdd/x_train.csv')
    y_train = pd.read_csv('processed/kdd/y_train.csv')
    norm = np.where(y_train['label'] == 0)[0]
    x_train = x_train.iloc[norm]
    print(x_train.shape)

print('start training....')
autoencoder = Autoencoder(x_train)
history = autoencoder.fit(
    x_train, x_train,
    batch_size=batch_size, epochs=num_epochs, validation_split=valid_rate, shuffle=shuffle)

autoencoder.save('results/' + dataset + '/' + model_filename + '.h5')
history_df = pd.DataFrame(history.history)
history_df.to_csv('results/' + dataset + '/training_history_' + model_filename + '.csv', index=False)
print('training completed')
# threshold = history.history["loss"][-1]
# print(threshold)
