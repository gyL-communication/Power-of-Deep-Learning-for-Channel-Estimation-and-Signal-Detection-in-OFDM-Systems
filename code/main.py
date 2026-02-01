from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

K = 64
CP = K // 4
P = 64  # number of pilot carriers per OFDM block
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
pilotCarriers = allCarriers[::K // P]  # Pilots is every (K/P)th carrier.
dataCarriers = np.delete(allCarriers, pilotCarriers)
mu = 2
payloadBits_per_OFDM = K * mu

SNRdb = 20  # signal to noise-ratio in dB at the receiver

Clipping_Flag = True #False

mapping_table = {
    (0, 0): -1 - 1j,
    (0, 1): -1 + 1j,
    (1, 0): 1 - 1j,
    (1, 1): 1 + 1j,
}

demapping_table = {v: k for k, v in mapping_table.items()}


def Clipping(x, CL):
    sigma = np.sqrt(np.mean(np.square(np.abs(x))))
    CL = CL * sigma
    x_clipped = x.copy()
    clipped_idx = abs(x_clipped) > CL
    x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx] * CL), abs(x_clipped[clipped_idx]))
    return x_clipped


def PAPR(x):
    Power = np.abs(x) ** 2
    PeakP = np.max(Power)
    AvgP = np.mean(Power)
    PAPR_dB = 10 * np.log10(PeakP / AvgP)
    return PAPR_dB


def Modulation(bits):
    bit_r = bits.reshape((int(len(bits) / mu), mu))
    return (2 * bit_r[:, 0] - 1) + 1j * (2 * bit_r[:, 1] - 1)


def OFDM_symbol(Data, pilot_flag):
    symbol = np.zeros(K, dtype=complex)
    symbol[pilotCarriers] = pilotValue
    symbol[dataCarriers] = Data
    return symbol


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(OFDM_time):
    cp = OFDM_time[-CP:]
    return np.hstack([cp, OFDM_time])


def channel(signal, channelResponse, SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved ** 2))
    sigma2 = signal_power * 10 ** (-SNRdb / 10)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*convolved.shape) + 1j * np.random.randn(*convolved.shape))
    return convolved + noise


def removeCP(signal):
    return signal[CP:(CP + K)]


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest


def get_payload(equalized):
    return equalized[dataCarriers]


def PS(bits):
    return bits.reshape((-1,))


def ofdm_simulate(codeword, channelResponse, SNRdb):
    OFDM_data = np.zeros(K, dtype=complex)
    OFDM_data[allCarriers] = pilotValue
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    if Clipping_Flag:
        OFDM_TX = Clipping(OFDM_TX, CR)
    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX)

    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword)
    symbol[np.arange(K)] = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    if Clipping_Flag:
        OFDM_withCP_cordword = Clipping(OFDM_withCP_cordword, CR)
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    #OFDM_RX_noCP_codeword = DFT(OFDM_RX_noCP_codeword)  ?don't really understand this step
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP))),
                           np.concatenate((np.real(OFDM_RX_noCP_codeword), np.imag(OFDM_RX_noCP_codeword))))), abs(
        channelResponse)


# load / generate pilot
Pilot_file_name = 'Pilot_' + str(P)
if os.path.isfile(Pilot_file_name):
    print('Load Training Pilots txt')
    bits = np.loadtxt(Pilot_file_name, delimiter=',')
else:
    bits = np.random.binomial(n=1, p=0.5, size=(K * mu,))
    np.savetxt(Pilot_file_name, bits, delimiter=',')

pilotValue = Modulation(bits)
CR = 1

class Encoder(nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output):
        super(Encoder, self).__init__()
        
        self.layer1 = nn.Linear(n_input, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.layer4 = nn.Linear(n_hidden_3, n_output)
      
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.1)
                nn.init.trunc_normal_(m.bias, std=0.1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.sigmoid(self.layer4(x))
        return x


def training():
    # Training parameters
    training_epochs = 1000#20000
    batch_size = 256
    display_step = 5
    test_step = 1000

    # Network Parameters
    n_hidden_1 = 500
    n_hidden_2 = 250
    n_hidden_3 = 120
    n_input = 256
    n_output = 16

    # model
    model = Encoder(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_output).to(device)
    
    # loss
    criterion = nn.MSELoss()
    
    # lr
    learning_rate_current = 0.001
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate_current)

    # The H information set
    H_folder = './H_dataset/'
    train_idx_low = 1
    train_idx_high = 301
    test_idx_low = 301
    test_idx_high = 401

    # Saving Channel conditions to a large matrix
    channel_response_set_train = []
    for train_idx in range(train_idx_low, train_idx_high):
        H_file = H_folder + str(train_idx) + '.txt'
        with open(H_file) as f:
            for line in f:
                numbers_str = line.split()
                numbers_float = [float(x) for x in numbers_str]
                h_response = np.asarray(numbers_float[0:int(len(numbers_float) / 2)]) + 1j * np.asarray(
                    numbers_float[int(len(numbers_float) / 2):len(numbers_float)])
                channel_response_set_train.append(h_response)

    channel_response_set_test = []
    for test_idx in range(test_idx_low, test_idx_high):
        H_file = H_folder + str(test_idx) + '.txt'
        with open(H_file) as f:
            for line in f:
                numbers_str = line.split()
                numbers_float = [float(x) for x in numbers_str]
                h_response = np.asarray(numbers_float[0:int(len(numbers_float) / 2)]) + 1j * np.asarray(
                    numbers_float[int(len(numbers_float) / 2):len(numbers_float)])
                channel_response_set_test.append(h_response)

    print('length of training channel response', len(channel_response_set_train),
          'length of testing channel response', len(channel_response_set_test))

    for epoch in range(training_epochs):
        print(epoch)
        
        # lr decay
        if epoch > 0 and epoch % 2000 == 0:
            learning_rate_current = learning_rate_current / 5
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_current

        avg_cost = 0.
        total_batch = 50

        model.train()
        for index_m in range(total_batch):
            input_samples = []
            input_labels = []
            for index_k in range(0, 1000):
                bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
                channel_response = channel_response_set_train[np.random.randint(0, len(channel_response_set_train))]
                signal_output, para = ofdm_simulate(bits, channel_response, SNRdb)
                input_labels.append(bits[16:32])
                input_samples.append(signal_output)

            batch_x = torch.tensor(np.asarray(input_samples), dtype=torch.float32).to(device)
            batch_y = torch.tensor(np.asarray(input_labels), dtype=torch.float32).to(device)

            # forward
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            
        
            loss.backward()
            optimizer.step()

            avg_cost += loss.item() / total_batch

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

            model.eval()
            with torch.no_grad():
                input_samples_test = []
                input_labels_test = []
                test_number = 1000

                if epoch % test_step == 0:
                    print("Big Test Set")
                    test_number = 10000

                for i in range(0, test_number):
                    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
                    channel_response = channel_response_set_test[np.random.randint(0, len(channel_response_set_test))]
                    signal_output, para = ofdm_simulate(bits, channel_response, SNRdb)
                    input_labels_test.append(bits[16:32])
                    input_samples_test.append(signal_output)

                batch_x_test = torch.tensor(np.asarray(input_samples_test), dtype=torch.float32).to(device)
                batch_y_test = torch.tensor(np.asarray(input_labels_test), dtype=torch.float32).to(device)

                y_pred_test = model(batch_x_test)
                mean_error = torch.mean(torch.abs(y_pred_test - batch_y_test))
                
                #  mean_error_rate
                pred_bits = torch.sign(y_pred_test - 0.5)
                true_bits = torch.sign(batch_y_test - 0.5)
                mean_error_rate = 1 - torch.mean((pred_bits == true_bits).float())

                print("OFDM Detection QAM output number is", n_output, "SNR =", SNRdb, "Num Pilot", P,
                      "prediction and the mean error on test set are:", mean_error.item(), mean_error_rate.item())
  
                # mean_error_rate_train
                batch_x_train = torch.tensor(np.asarray(input_samples), dtype=torch.float32).to(device)
                batch_y_train = torch.tensor(np.asarray(input_labels), dtype=torch.float32).to(device)
                y_pred_train = model(batch_x_train)
                mean_error_train = torch.mean(torch.abs(y_pred_train - batch_y_train))
                
                pred_bits_train = torch.sign(y_pred_train - 0.5)
                true_bits_train = torch.sign(batch_y_train - 0.5)
                mean_error_rate_train = 1 - torch.mean((pred_bits_train == true_bits_train).float())

                print("prediction and the mean error on train set are:", mean_error_train.item(), mean_error_rate_train.item())

    print("optimization finished")


if __name__ == '__main__':
    training()
