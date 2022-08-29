# simulation mesoLIF population
# membrane potential dynamics: taum*du/dt=-u+mu+taum*J*y
# synaptically filtered input: taus*dy/dt=-y+A(t-delay)
# here: M=1 population (but several excitatory and inhibitory populations possible)
# copyright Tilo Schwalger, August 2022


using PyPlot, Printf, HDF5

include("sim3.jl")

N=200
M =1
Nrecord = 1 # number populations to be recorded
#naiv=true
naiv=false

T = 0.3# simulation time (seconds)
dt_record = 0.001  #bin size of population activity in seconds
dt = 0.0005        #simulation time step in seconds
Nbin = round(Int, T/dt_record)
tt= dt_record * collect(1:Nbin)

seed=1          #seed for finite-size noise

J=0.0 #mV
taum =0.02 #seconds
taus = 0.0 #seconds



params = Dict("mu" => 20.,
              "hazard_c" => 10.0,      #Hz
              "hazard_Delta_u" => 1.0, #mV
              "vth" => 10.,    #mV
              "vreset" => 0.0,
              "taum_e" => taum,  #membrane time constant for exc. neurons (s)
              "taum_i" => taum,  #membrane time constant for inh. neurons (s)
              "delay" => 0.001,
              "tref" => 0.0,
              "taus_e" => taus, #synaptic time constant of exc. synapses (s)
              "taus_i" => taus, #synaptic time constant of inh. synapses (s)
              "M" => M,
              "Me" => M,
              "N" => N * ones(Int, M))

params["weights"] = J .* ones(M,M)



rate1, A1 = sim(T, dt, dt_record, params, Nrecord, seed)



figure(1)
clf()
plot(tt,A1[1,:])
plot(tt,rate1[1,:])
ylabel("rate [Hz]")
xlabel("time [s]")
subplots_adjust(top=0.88, bottom=0.235, left=0.235, right=0.9, hspace=0.2, wspace=0.2)
show()

