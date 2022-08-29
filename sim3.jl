# simulation of stochastic integral equation
# here constant external drive, but code can be easily modified to include time-dependent drive
# copyright Tilo Schwalger 2022

using Distributions, Random


function hazard(u, c, Delta_u, Vth)
    return c*exp((u-Vth)/Delta_u)             # hazard rate
end

function Pfire(u, c, Delta_u, Vth, lambda_old, dt)
    lambda = hazard(u, c, Delta_u, Vth)
    Plam = 0.5 * (lambda_old + lambda) * dt
    if (Plam>0.01)
        Plam = -expm1(-Plam)
    end
    return Plam, lambda
end

        
function sim(T, dt, dt_rec, params, Nrecord, seed)
    rng2 = MersenneTwister(seed) 

    M = params["M"]
    Me = params["Me"]
    Mi = M - Me
    N = params["N"]
    
    mu = params["mu"]
    Delta_u = params["hazard_Delta_u"]
    c = params["hazard_c"]
    vreset = params["vreset"]
    vth = params["vth"]
    tref = params["tref"]
    delay = params["delay"]
    n_ref = round(Int, tref/dt)
    n_delay = round(Int, delay/dt)
    
    #membrane time constants
    tau = zeros(Float16, M)
    dtau = zeros(Float16, M)
    
    tau[1:Me] .= params["taum_e"]
    tau[(1+Me):M] .= params["taum_i"]
    dtau[1:Me] .= dt/params["taum_e"]
    dtau[(1+Me):M] .= dt/params["taum_i"]

    #synaptic time constants
    Etaus = zeros(Float16, M)
    for m in 1:Me
        if params["taus_e"] > 0
            Etaus[m] = exp(-dt / params["taus_e"])
        end
    end
    for m in 1:Mi
        if params["taus_i"] > 0
            Etaus[Me+m] = exp(-dt / params["taus_i"])
        end
    end

    weights = params["weights"]

    #quantities to be recorded
    Nsteps = round(Int,T/dt)
    Nsteps_rec = round(Int,T/dt_rec)
    Nbin = round(Int, dt_rec/dt) #bin size for recording 
    Abar = zeros(Float32,(Nrecord, Nsteps_rec))
    A = zeros(Float32,(Nrecord, Nsteps_rec))

    #initialization
    L=zeros(Int, M)
    for i=1:M
        L[i] = round(Int, (5 * tau[i] + tref) / dt) + 1 #history length of population i
    end
    Lmax = maximum(L)
    S=ones(Float32, (M, Lmax))
    u=vreset * ones(Float32, (M, Lmax))
    n=zeros(Float32, (M, Lmax))
    lam=zeros((M, Lmax))
    x=zeros(Float32, M)
    y=zeros(Float32, M)
    z=zeros(Float32, M)
    for i=1:M
        n[i,L[i]] = 1. #all units fired synchronously at t=0
    end
        
    h = vreset * ones(M)
    lambdafree = zeros(M)
    for i=1:M
        lambdafree[i]=hazard(h[i], c, Delta_u, vth)
    end

    #begin main simulation loop
    for ti = 1:Nsteps
	if mod(ti,Nsteps/100) == 1  #print percent complete
	    @printf("\r%d%% ",round(Int,100*ti/Nsteps))
	end
    
	t = dt*ti
        i_rec = ceil(Int, ti/Nbin)

        synInput=zeros(M)
        for i=1:M
            for j=1:M
                synInput[i] += weights[i,j] * y[j]
            end
        end
        
	for i = 1:M
	    h[i] += dtau[i]*(mu-h[i]) + synInput[i] * dt
            Plam, lambdafree[i] = Pfire(h[i], c, Delta_u, vth, lambdafree[i], dt)
            W = Plam * x[i]
            X = x[i]
            Z = z[i]
            Y = Plam * z[i]
            z[i] = (1-Plam)^2 * z[i] + W
            x[i] -= W
            
            for l=2:L[i]-n_ref
                u[i, l-1] = u[i,l] + dtau[i] * (mu - u[i,l])  + synInput[i] * dt
                Plam, lam[i,l-1] = Pfire(u[i,l-1], c, Delta_u, vth, lam[i,l], dt)
                m = S[i, l] * n[i,l]
                v = (1 - S[i,l]) * m
                W += Plam * m
                X += m
                Y += Plam * v
                Z += v
                S[i,l-1] = (1 - Plam) * S[i, l]
                n[i,l-1] = n[i,l]
            end
            x[i] += S[i,1] * n[i, 1]
            z[i] += (1 - S[i,1]) *  S[i,1] * n[i, 1]
            for l=L[i]-n_ref+1:L[i]  #refractory period
                X+=n[i,l]
                n[i,l-1]=n[i,l]
            end
   
            if (Z>0)
                PLAM = Y/Z
            else
                PLAM = 0
            end
            
            nmean = max(0, W +PLAM * (1 - X))
            if nmean>1
                nmean = 1
            end

            distrib = Binomial(N[i], nmean)
            n[i, L[i]] = rand(rng2, distrib) / N[i] # population activity (fraction of neurons spiking)
            
            y[i] = y[i] * Etaus[i] + n[i, L[i] - n_delay] / dt * (1 - Etaus[i])

            if (i <= Nrecord)
                Abar[i, i_rec] += nmean
                A[i,i_rec] += n[i,L[i]]
            end #end loop over recorded neurons
            
	end #end loop over populations
    end     #end loop over time
    Abar /= (Nbin * dt)
    A  /= (Nbin * dt)

    @printf("\r")
    
    return Abar, A
end
