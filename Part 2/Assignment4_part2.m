inputFilename = 'novel.txt'; 
my_text = textToBinaryArray(inputFilename);
 
function my_text = textToBinaryArray(filename)
% Read the text file
try
fileID = fopen(filename, 'r');
textData = fread(fileID, '*char')'; % Read as a character array
fclose(fileID);
catch
error('Error reading the text file.');
end
    
% Convert characters to binary representations
my_text = [];
for i = 1:length(textData)
charBinary = dec2bin(textData(i), 7); % Convert to 7-bit binary
my_text = [my_text, charBinary];
end
end

my_text_new = splitToSingleBits(my_text);
 
function my_text_new = splitToSingleBits(my_text)
    my_text_new = [];
    
    % Iterate through the 7-bit elements and split into single bits
    for i = 1:7:length(my_text)
        sevenBits = my_text(i:i+6); % Extract a 7-bit element
        singleBits = sevenBits - '0'; % Convert to a numeric array of single bits
        my_text_new = [my_text_new, singleBits];
    end
end

h = lzEncoder(my_text_new);
 
function h = lzEncoder(input_binary)
    dictionary = containers.Map;
    next_code = 0;
    h = [];
    
    % Initialize the dictionary with single-bit entries
    for bit = 0:1
        dictionary(num2str(bit)) = next_code;
        next_code = next_code + 1;
    end
    
    current_code = '';
    
    for bit = input_binary
        current_code = [current_code, num2str(bit)];
        if ~isKey(dictionary, current_code)
            % Add the current code to the dictionary
            dictionary(current_code) = next_code;
            next_code = next_code + 1;
            
            % Output the previous code
            h = [h, dictionary(current_code(1:end-1))];
            
            % Reset the current code to the last bit
            current_code = num2str(bit);
        end
    end
    
    % Encode the last code
    if isKey(dictionary, current_code)
        h = [h, dictionary(current_code)];
    end
end

%producing the Test_0.txt
outfile = fopen('Test_0.txt','w');
fprintf(outfile,'%s\n' ,h);
fclose(outfile);

input_data = h;
e = hamming74Encoder(input_data);
 
function e = hamming74Encoder(input_data)
    % Calculate the number of bits to pad
    pad_length = mod(-length(input_data), 4);
    
    % Pad the input data with zeros if needed
    if pad_length > 0
        input_data = [input_data, zeros(1, pad_length)];
    end
    
    % Define the generator matrix for Hamming(7,4)
    G = [1 0 0 0 1 1 1;
         0 1 0 0 0 1 1;
         0 0 1 0 1 0 1;
         0 0 0 1 1 1 0];
    
    % Initialize the encoded stream
    e = [];
    
    % Encode each 4-bit block separately
    for i = 1:4:length(input_data)
        block = input_data(i:i+3); % Extract a 4-bit block
        encoded_block = mod(block * G, 2); % Encode the block using matrix multiplication
        e = [e, encoded_block];
    end
end
 
outfile = fopen('Text_1.txt','w');
fprintf(outfile,'%i\n',e);
fclose(outfile);
%16_QAM modulation operation
input_data = e; % Variable-length input data
modulated_signal = qam16Modulator(input_data);
disp('Modulated Signal (16-QAM):');
disp(modulated_signal);
 
%16_QAM demodulation operation
received_signal = modulated_signal; 
demodulated_data = qam16Demodulator(received_signal);
disp('Demodulated Data (16-QAM):');
disp(demodulated_data);
 
% hamming decorder operation
encoded_data_hamming = demodulated_data; 
decoded_data_hamming = hamming74Decoder(encoded_data_hamming);
disp('Decoded Data Stream (Hamming(7,4)):');
disp(decoded_data_hamming);

%-----------------------------------------

function modulated_signal = qam16Modulator(input_data)
% Ensure the length of input_data is valid
if mod(length(input_data), 4) ~= 0
warning('Input data length is not a multiple of 4. Padding with zeros.');
% Pad input_data with zeros to make it a multiple of 4
input_data = [input_data, zeros(1, 4 - mod(length(input_data), 4))];
end
 
% Define the 16-QAM constellation points
constellation = [-3 -3; -3 -1; -3 3; -3 1;
-1 -3; -1 -1; -1 3; -1 1;
3 -3;  3 -1;  3 3;  3 1;
1 -3;  1 -1;  1 3;  1 1];
 
% Initialize the modulated signal
modulated_signal = zeros(1, length(input_data)/4);
 
% Map each 4-bit input data block to constellation points
for i = 1:4:length(input_data)
% Convert the 4 bits to a decimal value
symbol_index = bi2de(input_data(i:i+3), 'left-msb');
modulated_signal((i+3)/4) = constellation(symbol_index + 1);
end
end

%-----------------------------------------

function demodulated_data = qam16Demodulator(received_signal)
% Define the 16-QAM constellation points
constellation = [-3 -3; -3 -1; -3 3; -3 1;
-1 -3; -1 -1; -1 3; -1 1;
3 -3;  3 -1;  3 3;  3 1;
1 -3;  1 -1;  1 3;  1 1];
 
% Initialize the demodulated data array
demodulated_data = zeros(1, length(received_signal) * 4);
 
% Demodulate the received signal
for i = 1:length(received_signal)
symbol = received_signal(i);
 
% Find the nearest constellation point to the received symbol
[~, index] = min(sum(abs(constellation - symbol).^2, 2));
 
% Convert the index to a 4-bit binary representation
binary_data = de2bi(index - 1, 4, 'left-msb');
 
% Append the binary data to the demodulated data array
demodulated_data((i - 1) * 4 + 1 : i * 4) = binary_data;
end
end
 
%-----------------------------------------

function decoded_data_hamming = hamming74Decoder(encoded_data_hamming)
% Define the parity-check matrix for Hamming(7,4)
H = [1 1 0 1 0 0 0;
1 0 1 1 0 0 0;
0 1 1 1 0 0 0;
1 1 1 0 1 0 0;
0 0 0 1 1 1 0;
0 0 0 0 1 1 1];
 
% Initialize the decoded data
decoded_data_hamming = [];
 
% Process full 7-bit blocks
for i = 1:7:length(encoded_data_hamming)-6
block = encoded_data_hamming(i:i+6); % Extract a 7-bit block
 
% Perform syndrome check to detect errors
syndrome = mod(block * H', 2);
 
% If syndrome is non-zero, correct the error if possible
if any(syndrome)
error_position = -1; % Initialize error_position to -1
 
% Manually search for the syndrome within the rows of H
for j = 1:size(H, 1)
if isequal(syndrome, H(j, :))
error_position = j;
break; % Exit the loop when a match is found
end
end
 
if error_position ~= -1
% Correct the error
block(error_position) = ~block(error_position);
else
% Cannot correct the error, treat it as uncorrectable
warning('Uncorrectable error in Hamming(7,4) code.');
end
end
 
% Extract the 4 data bits
decoded_block = block([3 5 6 7]);
 
% Append the decoded data to the result
decoded_data_hamming = [decoded_data_hamming, decoded_block];
end
 
% Process any remaining bits (less than a full 7-bit block)
if mod(length(encoded_data_hamming), 7) ~= 0
remaining_bits = encoded_data_hamming(i+7:end);
decoded_data_hamming = [decoded_data_hamming, remaining_bits];
end
end

% outputFilename_2 = 'Text_2.txt'; % Specify the output filename
% writeLZEncodedDataToFile(decoded_data_hamming, outputFilename_2);

% binary_input = modulated_signal;
% fs = 1000; %sampling fq
% tdm_signal = binaryTDMMultiplexer(binary_input, fs);
% disp("Binary TDM Multiplexing Completed");
%  
% function tdm_signal = binaryTDMMultiplexer(binary_input, fs);
%  
% t_binary = (0:1fs:1-1/fs);
% if length(binary_input)< length(t_binary)
%     
%     interpoated_signal = interp1(1:length(binary_input),binary_input)
%         inspace(1:length(binary_input),length(t_binary));
%  binary_input = interpolated_signal;
% end
%  
% %initiate TDM signal
% tdm_signal = binary_input;
% end

% M_values = [4, 16, 64];
% snr_dB = -10:1:20;
% num_simulations = 1e5;
% num_symbols = 1000;
%  
% ber = zeros(length(M_values), length(snr_dB));
%  
% for M_index = 1:length(M_values)
%     M = M_values(M_index);
%     for snr_index = 1:length(snr_dB)
%         snr = 10^(snr_dB(snr_index)/10);
%         error_count = 0;
%  
%         for j = 1:num_simulations
%             % Generate random QAM symbols
%             tx_symbols = qammod(randi([0 M-1], 1, num_symbols), M);
%  
%             % Calculate noise variance
%             noise_var = var(tx_symbols) / (snr * 2);
%  
%             % Generate AWGN with calculated noise variance
%             noise = sqrt(noise_var) * randn(1, num_symbols);
%  
%             % Add AWGN to the signal
%             rx_symbols = tx_symbols + noise;
%  
%             % Perform QAM demodulation
%             demodulated_symbols = qamdemod(rx_symbols, M);
%  
%             % Calculate symbol error rate
%             error_count = error_count + sum(demodulated_symbols ~= tx_symbols);
%         end
%  
%         ber(M_index, snr_index) = error_count / (num_simulations * num_symbols);
%     end
% end
% 

clc
clear all
close all

M = [2,4,8,16,32,64];
SNR=0:1:20;
SER = [];
for m=1:length(M)
ser_M = [];
for k=1:length(SNR)
yb = 10^(SNR(k)/10);
ser = erfc(sqrt(yb)*sin(pi/M(m)));
ser_M(k) = ser;
end
SER(m,:) = ser_M;
end
semilogy(SNR,SER(1,:),'r')
hold on
semilogy(SNR,SER(2,:),'k')
hold on
semilogy(SNR,SER(3,:),'m')
hold on
semilogy(SNR,SER(4,:),'c')
hold on
semilogy(SNR,SER(5,:),'g')
hold on
semilogy(SNR,SER(6,:))
hold on
axis([0 20 10^(-6) 1])

