function y = map(x, a, b)
    y = (x-min(x))*(b-a)/(max(x)-min(x)) + a;
end

