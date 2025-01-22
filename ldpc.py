import numpy as np


# Parameters
M = 8  # M-ARY PSK = 8 distinct symbols for encoding transmitted signal
Q = 4  # Q discrete states the phasor can take on the unit circle in the complex plane
T = 16  # Number of transmit antennas
R = 16  # Number of received signals
N = 10  # Initial length of codeword
K = 5   # Initial number of parity-check equations
bits = 1000  # Starting bits to transmit
snratio = 10  # Signal-to-noise ratio (in dB)

# manually make ldpc plus coding, encoding, and decoding functions 
def generate_parity_check_matrix(n,m, density = 0.3):
    # n = number of columns i.e codeword length 
    # m = number of rows i.e number of parity check equations 
    # density = proportion of non-zeros, ones 
    H = np.zeros((m,n), dtype=int)
    for i in range(m):
        ones_pos = np.random.choice(n, int(density*n), replace=False)
        H[i,ones_pos] = 1
    return H

def derive_generator_matrix(H):
    #given parity check matrix create generator matrix 
    m,n = H.shape
    message_bits = n-m 
    #gives [I | P^T]
    P = H[:,:message_bits]
    #returns form [I | P^T]
    G = np.hstack((np.eye(k,dtype=int),P.T))
    return G

def encode_message(G, message):
    #encode a binary message using generator matrix, mod 2
    message = np.array(message)
    codeword = np.dot(message, G) % 2 
    return codeword

def decoder(paritycheckmtx, recieved, max_iterations=50):
    #syndrome is a vector that represents difference between recieved codeword and nearest valid codeword
    #S = H • r^t (mod 2)
    decoded = recieved.copy()
    for _ in range(max_iterations):
        syndrome = np.dot(paritycheckmtx,decoded) % 2
        if np.count_nonzero(syndrome) == 0:
            break
        #identify bits to flip 
        error_pos = np.where(syndrome == 1)[0]
        for pos in error_pos:
            #flip
            decoded[pos] = 1 - decoded[pos]
    return decoded

#full ldpc pipeline
def ldpc(n,m,message,density = 0.3):
    #parity check matrix 
    H = generate_parity_check_matrix(n,m,density)
    print(f"Generated Parity Check Matrix H is:{H}")

    #generator matrix
    G = derive_generator_matrix(H)
    print(f"Generated Parity Check Matrix G is:{G}")

    #encode
    encoded_message = encode_message(G, message)
    print(f"Encoded message is:{encoded_message}")

    #noise
    recieved = encoded_message.copy()
    error_pos = np.randint(len(recieved))
    recieved[error_pos] = 1-recieved[error_pos]
    print(f"Recieved Codeword is:{recieved}")

    decoded = decoder(H, recieved)
    print(f"Decoded message is:{decoded}")


# Modulation: map bits to M-PSK symbols
def m_ary_psk_modulate(bits, M):
    symbols = []  # Holds modulated symbols (complex numbers)
    for i in range(0, len(bits), int(np.log2(M))):  # Process log2(M) bits at a time
        symbol_bits = bits[i:i+int(np.log2(M))]
        symbol = int("".join(str(b) for b in symbol_bits), 2)  # Binary to decimal
        phase = 2 * np.pi * symbol / M  # Calculate phase based on symbol index
        symbols.append(np.exp(1j * phase))  # Create complex modulated signal
    return np.array(symbols)


# Additive White Gaussian Noise (AWGN) Channel
def awgn_channel(signal, SNR_dB):
    noise_power = 10 ** (-SNR_dB / 10)
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise



# M-ary PSK Demodulation
def m_ary_psk_demodulate(received_signal, M):
    demodulated_bits = []
    for symbol in received_signal:
        phase = np.angle(symbol)  # Calculate phase of received signal
        symbol_idx = int(np.round((phase / (2 * np.pi)) * M) % M)  # Determine symbol index
        bits = np.array(list(np.binary_repr(symbol_idx, width=int(np.log2(M)))))  # Convert to binary
        demodulated_bits.extend(bits.astype(int))
    return np.array(demodulated_bits)



# test
message_length = N - K
message = np.random.randint(0, 2, message_length)
encoded_message, received_message, decoded_message = ldpc(N, K, message)

# Modulate codeword using M-PSK
modulated_signal = m_ary_psk_modulate(encoded_message, M)
noisy_signal = awgn_channel(modulated_signal, snratio)

# Demodulate received signal
demodulated_bits = m_ary_psk_demodulate(noisy_signal, M)
print(f"Demodulated bits:\n{demodulated_bits}")



#need to create AWGN to simulate background interferance
#need to add Q state oscillators as discrete values where Q=4 in order to decode transmitted signal
##need to add M-ARY PSK in order to encode transmitted signal
#need to make create H matrix for resonator network
