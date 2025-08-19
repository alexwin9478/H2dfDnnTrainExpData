function [s_norm, s_min, s_range] = normalize_data(s)
s_min = min(s);
s_range = max(s) - min(s);
s_norm = (s - min(s))./ s_range;

% mu = min(data);
% sig = max(data);
% data_n = ((data - mu) / (sig-mu));
end

