function price = price_vg(spot, strike, maturity, r, ...
    sigma, nu, theta, alpha, delta, p)
%Price Call option with Variance Gamma exp-Levy model

    %   Default values for the arguments p, delta and alpha
    if nargin < 8
        alpha = 1.5;
    end
    if nargin < 9
        delta = 0.25;
    end
    if nargin < 10
        p = 12;
    end
    
    %number of points in grid of strikes
    N = 2^p;
    k_step = (2 * pi)/(N * delta);
    
    % u_m = m*delta, m = 0, ..., N-1
    u = 0 : delta : ((N - 1) * delta); 
    % k_n = - theta/2 + n*k_step, n = 0, ..., N-1
    k = ((-k_step * N) / 2):k_step:((k_step * (N-1)) / 2);
    
    %   x is the argument that we will apply FFT to
    F = VarianceGamma(u - (alpha + 1) * 1i, log(spot), maturity, r, ...
        sigma, nu, theta);
    F = F ./ (alpha ^ 2 + alpha - u .^ 2 + 1i * (2 * alpha + 1) * u);
    F = exp(-r * maturity) * delta * F .* exp(1i * (k_step * N / 2) * u);
    %   apply simpson's rule to x
    F = (F / 3) .* (3 + (-1).^(1:N) - ((0:(N - 1)) == 0));
    %   apply FFT and get call prices
    C_k = real((exp(-alpha * k) / pi) .* fft(F));
    
    %   find range of indexes [min_i, max_i] that contain all log(strikes)
    %   (+/-2 on each side for cubic and spline interpolation modification)
    logK = log(strike);
    min_i = floor((min(logK) + (k_step * N) / 2 ) / k_step + 1) - 2;
    max_i = ceil((max(logK) + (k_step * N) / 2 ) / k_step + 1) + 2;
    x = k(min_i:1:max_i);
    y = C_k(min_i:1:max_i);
    
    %   interpolate values of log(strike) 
    price = interp1(x, y, logK);
end

function cf = VarianceGamma(u, lnS, T, r, sigma, nu, theta)
%Characteristic function of V.G. process with martingale correction
	omega = (1 / nu) * log(1 - theta * nu - 0.5 * (sigma^2) * nu);
    exponent = 1i * u * (lnS + (r + omega) * T) - (T / nu) * ...
        log(1 - 1i * theta * nu * u + 0.5 * sigma^2 * nu * (u .^ 2));
    cf = exp(exponent);
end

