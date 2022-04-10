

####### Simulating the histogram of test statistic x.a.bar #####
### (1) First under the null hypothesis H0: theta = 1
n = 10
#n=2
x.a.bar.1 =c()
x.g.bar.1 =c()
x.max.1 = c()
simulation.number = 100000
for (i in 1: simulation.number){
  x.n = runif(n)  #shape1 =1, shape2 = 1 means uniform
  x.am = mean(x.n)
  x.a.bar.1 = c(x.a.bar.1,x.am)
  x.gm = exp(mean(log(x.n)))
  x.g.bar.1 = c(x.g.bar.1,x.gm)
  x.max = max(x.n)
  x.max.1 = c(x.max.1,x.max)
}
q95.a = quantile(x.a.bar.1, .95)
q95.a   # q95.a.10 = 0.6492847 
q95.g = quantile(x.g.bar.1, .95)
q95.g   # # q95.g.10 = 0.5794109 
q95.max = quantile(x.max.1, .95)
q95.max
theta = seq(1.1, 3, .1)
simu.size = 100000
E2P.a =c()
E2P.g =c()
E2P.m =c()
for(j in 1:length(theta)){
   root = 1/theta[j]
   x.a.bar =c()
   x.g.bar =c()
   x.max = c()
  for (i in 1: simu.size){
    x.n = rbeta(n, shape1 = theta[j], shape2 =1) 
## In my demonstration, I replaced the previous line by x.n = (runif(n))^root  
    x.am = mean(x.n)
    x.a.bar = c(x.a.bar,x.am)
    x.gm = exp(mean(log(x.n)))
    x.g.bar = c(x.g.bar,x.gm)
    largest = max(x.n)
    x.max = c( x.max, largest)
  }
   E2P.a = c(E2P.a,  sum(x.a.bar < q95.a)/simu.size)
   E2P.g = c(E2P.g,  sum(x.g.bar < q95.g)/simu.size)
   E2P.m = c(E2P.m, sum(x.max < q95.max)/simu.size)
}
E2P.a.10=E2P.a
E2P.g.10=E2P.g
E2P.m.10 =E2P.m 

power.true.m = .95^theta



### Plot of Type 2 Error probabilities 
par(mfrow=c(1,1))
plot(theta, E2P.g.10, main="Uniform example:Type 2 Error probabilities",
     xlab=expression(theta),
     ylab="Type 2 Error Probs", ylim = c(0,1), type="l", lty =5, col = 3,
     col.main="red", col.lab="blue", col.sub="black")

points(theta,power.true.m, type="l", lty =2, col =6)
points(theta,E2P.m.10, type="l", lty =7, col =4)
points(theta,E2P.a.10, type="l", lty =4, col =2)

legend(1, .1, legend=c("am", "gm", "max"),
       col=c(2, 3,4), lty=c(4,5,7), cex=0.8)
legend(2.5,.8, legend = "EP1 = 0.05", bty ="n")
legend(2.5,.6, legend = "n=10", bty ="n")


############# Mar 22, 2022. The codes below computed the pdf 
### f(u) = theta*u^{theta -1} for the null hypo theta = 1 (in yellow), and for the alternative
# theta > 1, three separate alternatives, namely, theta = 1.5 (in red), 
# theta = 2.5 (in green), theta = 3 (in blue)
#########
u = seq(.01,1,.05)
f1 =rep(1,length(u))
f1.5 = 1.5*u^(1.5-1)
f2.5 = 2.5*u^(2.5-1)
f3.0 =3*u^(3-1)
plot(u,f1.5, ylim = c(0,3), xlim = c(0,1), type ="l", lty = 2, lwd = 5, col=2, 
     main ="Null and Three pdfs under three alternatives")
points(u,f2.5, type ="l", lty = 3, lwd = 5, col=3)
points(u,f3.0, type ="l", lty = 5, lwd = 5, col=4)
points(u,f1,type ="l", lty = 6, lwd = 2, col=7)
legend(.2, 2.5, legend=c("1","1.5", "2.5", "3.0"),
       col=c(7,2, 3,4), lty=c(6,2,3,5), cex=0.8)


