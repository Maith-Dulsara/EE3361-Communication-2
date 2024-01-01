load handel.mat
filename = 'handel.wav';
audiowrite(filename,y,Fs);
[y,Fs] = audioread('handel.wav');

[N,~] = size(y);
ts = 1/Fs;
tmax = (N-1)*ts;
time = 0: ts: tmax; %time array

% Output time domain signal
figure(1) 
plot(time,y)
xlabel('Time'); ylabel('Amplitude');
title('Time domain signal');

f = -Fs/2: Fs/(N-1): Fs/2;      % Generating the freq array
freq = fftshift(fft(y));        % obtaining each freq by fft function
figure(2) % Output frequency domain signal
plot(f,abs(freq))
xlabel('Frequency'); ylabel('Power');
title('Frequency domain spectrum');

% Generate a unique source signal---------------------------------------
SNR = 0.0004*(9)^2 - 0.02*9 + 0.25; % Index = 19/ENG/009
z = awgn( y, SNR,'measured');

audiowrite('audio_1.wav', z, Fs); % Save the audio

[N,P] = size(z);
ts = 1/Fs;
tmax = (N-1)*ts;
time = 0: ts: tmax;

figure(3) % Output time domain signal 
plot(time, z)
xlabel('Time'); ylabel('Amplitude');
title('Time domain signal');

f = -Fs/2: Fs/(N-1): Fs/2;
freq1 = fftshift(fft(z)); 

figure(4) % Output frequency domain signal
plot(f, abs(freq1))
xlabel('Frequency'); ylabel('Power');
title('Frequency domain spectrum');


%Sampling the signal----------------------------------------------------
[N,~] = size(z);
s = zeros(1, N*2-1);  % Sampling freq = 2*Fs (2*N)                                            

for i = (1:2*N-1)
    if (mod(i,2)==0)
        s(i) = (z(i/2) + z(i/2+1))/2;   
    else 
        s(i) = z(ceil(i/2));
    end
end

figure(5)
stem(s);
xlabel('n'); ylabel('magnitude');
title('Sampled signal');

%Quantization of the sampled signal-------------------------------------
quantization_level = 6;
[q, q_error] = Quantization(s, quantization_level);

figure(6)
stairs(q);
xlabel('n'); ylabel('Magnitude');
title('Quantized signal');

audiowrite('audio_2.wav', q , 2*Fs);

% Encoding of the quantized signal--------------------------------------
vmax = max(s);  % Defining the max and min values of the signal
vmin = min(s);

b = bitmapping(q, quantization_level, vmax, vmin);

% Modulating the quantized signal using 64-QAM--------------------------
modOrder = 64;
m = QAMmodulator (b, modOrder, true);

% Plot the constallation diagram
scatterplot(m);
xlabel('In phase');
ylabel('Quadrature');
title('64-QAM, Average Power = 1 W');

% Transmission through the channel--------------------------------------
seed = 9;    % index = 009
snr = 10;    
c = awgn(m,snr, 'measured',seed); 

% To find the precentage errors
Noise_error = pctError(c, m);

% Vary the SNR by 2 intervals to plot constellation diagram
for snr = 1:2:10
    
    c = awgn(m,snr, 'measured',seed);
    scatterplot(c);
    xlabel('In phase');
    ylabel('Quadrature');
    str = sprintf('64-QAM, SNR value = %d', snr); 
    title(str)
        
end

% Plot SNR vs error graph
SNR = [1:2:10];
errorPlot = zeros(1,5);

for i = 1:2:10
    c = awgn(m,i, 'measured',seed);
    
    if i == 1
        errorPlot(i) = pctError(c, m);  
    elseif i == 3
        errorPlot(i-1) = pctError(c, m);
    elseif i == 5
        errorPlot(i-2) = pctError(c, m);
    elseif i == 7
        errorPlot(i-3) = pctError(c, m);
    elseif i == 9
        errorPlot(i-4) = pctError(c, m);
    end   
end


figure(13)
plot(SNR, errorPlot, '-o')
grid on;
xlabel('SNR value');
ylabel('Percentage Error');
title('SNR vs error plot')

% Obtain the received signal r over this channel-----------------------
r = awgn( m, 20, 'measured',seed);

% Constellation Plot
scatterplot(r); 
xlabel('In phase');
ylabel('Quadrature');
title('64-QAM, SNR value = 20'); 

% Find the percentage error by pctError function
errorRX = pctError(r, m);

% Demodulation of r signal----------------------------------------------
modOrder = 64;
d = DemodQAM (r,modOrder);

% precetange error after demodulation
errorAfterMod = pctErrorOfBitArray(d,b);

% Perfom reverse bitmapping--------------------------------------------
t = revBitMapping(d, quantization_level, vmax, vmin);

% save the audio as audio_3.
audiowrite('audio_3.wav', t , 2*Fs)

% Obtain the time-domain representation
figure(15)
stem(t)
xlabel('Time'); 
ylabel('Amplitude');
title('Time domain representation of the received signal');

% Used Functions-------------------------------------------------------

% Perfom reverse bitmapping
function t = revBitMapping(d, level, vmax, vmin)

    % converting the bit stream d to quantized values 
    B = reshape(d,level,[]); B = B';   % Dividing into 6 bit segments 
    B = char(48+B);         % Converting to char for bin2dec function
    q_index = bin2dec(B);   % Perfom bin2dec operation
    q_index = q_index';     % Convert the column matrix into array

    q_levels = 2^level;                 % no of levels
    lsb = (vmax- vmin)/((q_levels)-1);  % size between two levels
    t = q_index*lsb + vmin;         % Reversing the quantization

end

% Demodulation of r signal
function d = DemodQAM (RXsig, M)

    % Defining a zero array for complex numbers
    Dsig = complex(zeros(1,length(RXsig)));
    
    KMOD = 1/sqrt(42); % KMOD value for 64-QAM is 1/SQRT(42).
    RXsig = RXsig/KMOD; % Converting back to the original mapped values
    
    N = sqrt(M)-1; % 7
    % Checking each range to assign points to the points in constellation
    % Eg: points between 6<x<4 = 5
    
    for i = 1:length(RXsig)
        
        % real part
        found = false;
        for x = -N+1: 2: N-1  % check boundary values -6 to 6
            if real(RXsig(i)) < x
                Dsig(i) = x-1;
                found = true;
                break;
            end  
        end
       
        if ~found
           Dsig(i) = N; % check >6 values
        end
        
        % imaginary part
        found = false;
        for x = -N+1:2:N-1
            if imag(RXsig(i)) < x
                Dsig(i) = Dsig(i) + j*(x-1);
                found = true;
                break;
            end  
        end
        
        if ~found
           Dsig(i) = Dsig(i) + j*N;
        end      
    end
    
    % Convert it into a bit stream
    % Defining the mapping table
    mapTable(1:8) = -7; mapTable(9:16) = -5;
    mapTable(17:24) = -1; mapTable(25:32) = -3;
    mapTable(33:40) = +7; mapTable(41:48) = +5;
    mapTable(49:56) = +1; mapTable(57:64) = +3;
    
    for i = 0:63
        if mod(i,8) == 0 
            mapTable(i+1) = mapTable(i+1)-7j;
        elseif mod(i+1,8) == 0
            mapTable(i+1) = mapTable(i+1)+3j;
        elseif mod(i+2,8) == 0
            mapTable(i+1) = mapTable(i+1)+1j;
        elseif mod(i+3,8) == 0
            mapTable(i+1) = mapTable(i+1)+5j;
        elseif mod(i+4,8) == 0
            mapTable(i+1) = mapTable(i+1)+7j;
        elseif mod(i+5,8) == 0
            mapTable(i+1) = mapTable(i+1)-3j;
        elseif mod(i+6,8) == 0
            mapTable(i+1) = mapTable(i+1)-1j;
        elseif mod(i+7,8) == 0
            mapTable(i+1) = mapTable(i+1)-5j;
        end
    end
    
     % Defining a zero array for decimal index in the mapping table
     Dsig_index = zeros(1,length(Dsig));
     
     % Assigning index values by checking each mapping table points
     for x=1:length(Dsig)
         for y=1:length(mapTable)
             if Dsig(x) == mapTable(y) 
             Dsig_index(x) = y-1; 
             end
         end
     end  
    
    % converting to a binary array
    d_mat = (dec2bin(Dsig_index,6)- '0'); 
    d = d_mat';
    d = d(:)';   
              
end   

% Function to find the precentage error between two complex signals
function error = pctError(sig1, sig2)
    % Calculating the mean value of error array
   for i = 1: length(sig1)
    error = 100*mean(abs(sig1(i)-sig2(i))/abs(sig2(i)));
   end
      
end

% Function to find the precentage error between two binary arrays
function error = pctErrorOfBitArray(sig1, sig2)
     count = 0;
     %counting through the elements looking for differences
     for x = 1:1:length(sig1)
        if sig2(x) ~= sig1(x)
        count = count + 1;
        end
     end
     %calculating the error percentage
     error = count*100/length(sig1);
    
end

% Function for 64-QAM modulator
function m = QAMmodulator(bsig, M, avgPower)
    k = log2(M); % to get the number of bits in a binary symbol
    num = length(bsig)/k; % Number of symbols
        
    % Define mapping table using Gray Mapping 
    % Assuming point 1 = 000 000 (-7-7j) and goes on
    mapTable(1:8) = -7;    mapTable(9:16) = -5;
    mapTable(17:24) = -1;  mapTable(25:32) = -3;
    mapTable(33:40) = +7;  mapTable(41:48) = +5;
    mapTable(49:56) = +1;  mapTable(57:64) = +3;
    
    for i = 0:63
        if mod(i,8) == 0
            mapTable(i+1) = mapTable(i+1)-7j;
        elseif mod(i+1,8) == 0
            mapTable(i+1) = mapTable(i+1)+3j;
        elseif mod(i+2,8) == 0
            mapTable(i+1) = mapTable(i+1)+1j;
        elseif mod(i+3,8) == 0
            mapTable(i+1) = mapTable(i+1)+5j;
        elseif mod(i+4,8) == 0
            mapTable(i+1) = mapTable(i+1)+7j;
        elseif mod(i+5,8) == 0
            mapTable(i+1) = mapTable(i+1)-3j;
        elseif mod(i+6,8) == 0
            mapTable(i+1) = mapTable(i+1)-1j;
        elseif mod(i+7,8) == 0
            mapTable(i+1) = mapTable(i+1)-5j;
        end
    end
    
    symbols = zeros(1, num); % Zero array for mappedSymbols
    
    % Map binary 'b' array into symbols
    for i = 1:k:length(bsig)
        bin_symbol = bsig(i: i+k-1); % Map to 6 bit binary symbol
        dec_value =  2^5 * bin_symbol(1) + 2^4 * bin_symbol(2)...
                     + 2^3 * bin_symbol(3) + 2^2 * bin_symbol(4) ...
                     + 2^1 * bin_symbol(5) + 2^0 * bin_symbol(6);
    
    % Mapping to the complex array
    symbols((i-1)/k+1) = mapTable(dec_value+1); 
    end
    
    %Construction of the signal
    m1 = symbols; 
    
    % Modify the function to have a Unit power output signal    
    KMOD = 1/sqrt(42); % KMOD value for 64-QAM is 1/SQRT(42).
    
    if avgPower == true
       m = m1*(KMOD); 
    else
       m = m1;       
    end
    
end

% Function for encoding the quantized signal
function b = bitmapping(q_sig, level, vmax, vmin)
    
    q_levels = 2^level;                 % no of levels
    lsb = (vmax- vmin)/((q_levels)-1);  % size between two levels   
    
    val = (q_sig- vmin)/lsb;
    b_mat = (dec2bin(val,6)- '0'); % gives the bit values as a double matrix
    
    b = b_mat';  % convert it to an array
    b = b(:)';
    
end

% Function for Quantizing the sampled signal
function [q, q_error] = Quantization(sig, level)

    vmax = max(sig);       %get upper limit 
    vmin = min(sig);       %get lower limit 
    
    % Calculate the dist between two levels
    lsb = (vmax-vmin)/((2^level)-1);  
    levels = vmin:lsb:vmax;         % generate level vector 
    q_levels = 2^level;             % No of leveles
    [~, index] = size(sig);         % getting the size

    q = zeros(1, index); % Generating a zero vector 
    q_error = zeros(1,index); % Generating a zero vector for error

    for i = 1:index    
        for j = 2:q_levels                                                    
            if (sig(i) < levels(j))  % Obtain rest of q values
                q(i) = levels(j-1);
                q_error(i) = sig(i) - q(i); % Calcualting the error
                break
            end
        end   
    end 

end



