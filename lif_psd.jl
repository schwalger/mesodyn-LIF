# simulation of recurrent net of mesoLIF population
# membrane potential dynamics: taum*du/dt=-u+mu+taum*J*y
# synaptically filtered input: taus*dy/dt=-y+A(t-delay)
# here: M=1 population (but several excitatory and inhibitory populations possible)
# August 2022 Tilo Schwalger


using PyPlot, DSP

include("sim3.jl")

N=200
M =1
Nrecord = 1 # number populations to be recorded

T = 5.# simulation time (s)
dt_record = 0.001  #bin size of population activity in s
dt = 0.0002        #simulation time step in s
Nbin = round(Int, T/dt_record)
tt= dt_record * collect(1:Nbin)

seed=1          #seed for finite-size noise

J=0.0 #mV
taum =0.02
taus = 0.0

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

params["weights"] = J .* zeros(1,1)

figure(1)
clf()
@time begin
    rate, A = sim(T, dt, dt_record, params, Nrecord, seed)
end


plot(tt,rate[1,:])
ylabel("rate [Hz]")
xlabel("time [s]")


df=1.0
nseg=round(Int, 1.0/(dt_record * df))
transient=round(Int, 1.0/dt_record)
x=A[transient:end] .- mean(A[transient:end])
y=welch_pgram(x,nseg,0,fs=1.0/dt_record)

figure(2,figsize=[3.75,2.8125])
clf()
plot(y.freq,0.5*y.power,color="tab:blue",label="mesoscopic model")
xlim((df,250))
xlabel(L"$f$ [Hz]")
ylabel("power spectrum [Hz]")
legend(loc=0)
subplots_adjust(top=0.91,bottom=0.165,left=0.17,right=0.95,hspace=0.2,wspace=0.2)
show()
