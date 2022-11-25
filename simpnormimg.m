function [I, low, gap] = simpnormimg(G)

lb = min(G(:));
ub = max(G(:));

gap = ub-lb;
low = lb;
I = (G - low) ./ gap;

end
