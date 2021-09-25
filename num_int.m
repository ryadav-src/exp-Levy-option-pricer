function sol = num_int(alpha, S0, K, r, T, theta, sigma, nu)
%Numerical integration with with Variance Gamma exp-Levy model
    k = log(K);
    lnS = log(S0);
    f = @(u) real(func(u, k, alpha, lnS, r, T, theta, sigma, nu));
    sol = (exp(-alpha*k - r * T ))/(pi).* ...
        integral(f, 0, Inf, 'AbsTol', 1e-16);
end

function f = func(u, k, alpha, lnS, r, T, theta, sigma, nu)
%integrand for forward FT that is known analytically
    cf = exp(- 1i*u*k).*VarianceGamma(u-(alpha+1)*1i, lnS, T, r, sigma, nu, theta);
    f = cf ./(alpha .^ 2 + alpha - u .^ 2 + 1i .* (2 * alpha + 1) .* u);
end

function cf = VarianceGamma(u, lnS, T, r, sigma, nu, theta)
%Characteristic function of V.G. process with martingale correction
	omega = (1 / nu) * log(1 - theta * nu - 0.5 * (sigma^2) * nu);
    exponent = 1i * u * (lnS + (r + omega) * T) - (T / nu) * ...
        log(1 - 1i * theta * nu * u + 0.5 * sigma^2 * nu * (u .^ 2));
    cf = exp(exponent);
end