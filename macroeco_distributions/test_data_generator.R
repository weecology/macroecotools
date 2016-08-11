library(VGAM)

out_dir = 'test_data.csv'
# Test values for Poisson lognormal are chosen from Table 1 and Table 2
# in Grundy Biometrika 38:427-434.
# In Table 1 the values are deducted from 1 which give p(0).
pln_pmf_table1 = matrix(c(.9749, -2, 2, 
                         .9022, -2, 8, 
                         .8317, -2, 16, 
                         .1792, .5, 2, 
                         .2908, .5, 8, 
                         .3416, .5, 16, 
                         .0000, 3, 2, 
                         .0069, 3, 8, 
                         .0365, 3, 16), byrow = T, ncol = 3)
# x = 0, untruncated
pln_pmf_table1 = t(apply(pln_pmf_table1, 1, function(x) c('pln', 'pmf', 0, x, 0)))
write.table(pln_pmf_table1, out_dir, row.names = F, quote = F, col.names = F, sep = ',')

pln_pmf_table2 = matrix(c(.0234, -2, 2, 
                          .0538, -2, 8, 
                          .0593, -2, 16, 
                          .1512, .5, 2, 
                          .1123, .5, 8, 
                          .0879, .5, 16, 
                          .0000, 3, 2, 
                          .0065, 3, 8, 
                          .0193, 3, 16), byrow = T, ncol = 3)
# x = 1, untruncated
pln_pmf_table2 = t(apply(pln_pmf_table2, 1, function(x) c('pln', 'pmf', 1, x, 0)))
write.table(pln_pmf_table2, out_dir, row.names = F, quote = F, col.names = F, 
            sep = ',', append = T)

# The following test data are generated using R
# 1. upper-truncated logseries
trunc_logser_pars = matrix(c(1, 0.1, 10,
                             2, 0.3, 5,
                             3, 0.9, 20), byrow = T, ncol = 3)
for (i in 1:nrow(trunc_logser_pars)){
  row = trunc_logser_pars[i, ]
  pmf = dlog(row[1], row[2]) / plog(row[3], row[2])
  write.table(t(c('trunc_logser', 'pmf', row[1], pmf, row[2:3])), out_dir, 
              row.names = F, quote = F, col.names = F, sep = ',', append = T)
  cdf = plog(row[1], row[2]) / plog(row[3], row[2])
  write.table(t(c('trunc_logser', 'cdf', row[1], cdf, row[2:3])), out_dir, 
              row.names = F, quote = F, col.names = F, sep = ',', append = T)
}

# 2. lower-truncated exponential
trunc_expon_pars = matrix(c(2, 0.1, 1,
                            3, 0.5, 0.2, 
                            4, 0.7, 0.4), byrow = T, ncol = 3)
for (i in 1:nrow(trunc_expon_pars)){
  row = trunc_expon_pars[i, ]
  pdf = dexp(row[1], row[2]) / (1 - pexp(row[3], row[2]))
  write.table(t(c('trunc_expon', 'pdf', row[1], pdf, row[2:3])), out_dir, 
              row.names = F, quote = F, col.names = F, sep = ',', append = T)
  cdf = (pexp(row[1], row[2]) - pexp(row[3], row[2])) / (1 - pexp(row[3], row[2]))
  write.table(t(c('trunc_expon', 'cdf', row[1], cdf, row[2:3])), out_dir, 
              row.names = F, quote = F, col.names = F, sep = ',', append = T)
}

# 3. lower-truncated pareto
trunc_pareto_pars = matrix(c(2, 1, 1, 
                             3, 2, 0.2, 
                             4, 3, 0.7), byrow = T, ncol = 3)
for (i in 1:nrow(trunc_expon_pars)){
  row = trunc_pareto_pars[i, ]
  pdf = dpareto(row[1], row[3], row[2])
  write.table(t(c('trunc_pareto', 'pdf', row[1], pdf, row[2:3])), out_dir, 
              row.names = F, quote = F, col.names = F, sep = ',', append = T)
  cdf = ppareto(row[1], row[3], row[2])
  write.table(t(c('trunc_pareto', 'cdf', row[1], cdf, row[2:3])), out_dir, 
              row.names = F, quote = F, col.names = F, sep = ',', append = T)
}

# 4. upper-truncated geometric with zeros
trunc_geom_zero_pars = matrix(c(1, 0.1, 10,
                           2, 0.3, 5,
                           3, 0.7, 20), byrow = T, ncol = 3)
for (i in 1:nrow(trunc_geom_zero_pars)){
  row = trunc_geom_zero_pars[i, ]
  pmf = dgeom(row[1], row[2]) / pgeom(row[3], row[2])
  write.table(t(c('trunc_geom_zeros', 'pmf', row[1], pmf, row[2:3])), out_dir, 
              row.names = F, quote = F, col.names = F, sep = ',', append = T)
  cdf = pgeom(row[1], row[2]) / pgeom(row[3], row[2])
  write.table(t(c('trunc_geom_zeros', 'cdf', row[1], cdf, row[2:3])), out_dir, 
              row.names = F, quote = F, col.names = F, sep = ',', append = T)
}

# 5. upper-truncated geometric without zeros
trunc_geom_pars = trunc_geom_zero_pars
for (i in 1:nrow(trunc_geom_pars)){
  row = trunc_geom_pars[i, ]
  pmf = dgeom(row[1], row[2])/ (pgeom(row[3], row[2]) - dgeom(0, row[2]))
  write.table(t(c('trunc_geom', 'pmf', row[1], pmf, row[2:3])), out_dir, 
              row.names = F, quote = F, col.names = F, sep = ',', append = T)
  cdf = (pgeom(row[1], row[2]) - dgeom(0, row[2])) / (pgeom(row[3], row[2]) - 
                                                      dgeom(0, row[2]))
  write.table(t(c('trunc_geom', 'cdf', row[1], cdf, row[2:3])), out_dir, 
              row.names = F, quote = F, col.names = F, sep = ',', append = T)
}

# 6. negative binomial without zeros (ie. lower-truncated)
trunc_nbinom_pars = matrix(c(20, 10, 0.2,
                             2, 8, 0.3, 
                             5, 20, 0.8), byrow = T, ncol = 3)
for (i in 1:nrow(trunc_nbinom_pars)){
  row = trunc_nbinom_pars[i, ]
  pmf = dnbinom(row[1], row[2], row[3])/ (1 - dnbinom(0, row[2], row[3]))
  write.table(t(c('trunc_nbinom', 'pmf', row[1], pmf, row[2:3])), out_dir, 
              row.names = F, quote = F, col.names = F, sep = ',', append = T)
  cdf = (pnbinom(row[1], row[2], row[3]) - pnbinom(0, row[2], row[3])) / 
        (1 - dnbinom(0, row[2], row[3]))
  write.table(t(c('trunc_nbinom', 'cdf', row[1], cdf, row[2:3])), out_dir, 
              row.names = F, quote = F, col.names = F, sep = ',', append = T)
}

# 7. lower-truncated weibull
trunc_weibull_pars = matrix(c(3, 1, 1, 2,
                              1.7, 5, 2, 1, 
                              2, 3, 4, 0.7), byrow = T, ncol = 4)
for (i in 1:nrow(trunc_weibull_pars)){
  row = trunc_weibull_pars[i, ]
  pdf = dweibull(row[1], row[2], row[3])/ (1 - pweibull(row[4], row[2], row[3]))
  write.table(t(c('trunc_weibull', 'pdf', row[1], pdf, row[2:4])), out_dir, 
              row.names = F, quote = F, col.names = F, sep = ',', append = T)
  cdf = (pweibull(row[1], row[2], row[3]) - pweibull(row[4], row[2], row[3])) / 
        (1 - pweibull(row[4], row[2], row[3]))
  write.table(t(c('trunc_weibull', 'cdf', row[1], cdf, row[2:4])), out_dir, 
              row.names = F, quote = F, col.names = F, sep = ',', append = T)
}
