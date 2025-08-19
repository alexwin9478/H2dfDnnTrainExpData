function [data_n, mu, sig] = dataTrainStandardized(data)
mu = mean(data);
sig = std(data);

data_n = (data - mu) / sig;

% mu = min(data);
% sig = max(data);
% data_n = ((data - mu) / (sig-mu));
end

