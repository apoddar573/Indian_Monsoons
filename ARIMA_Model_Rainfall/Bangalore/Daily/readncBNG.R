library(ncdf4)
library(chron)

#reding file

open <-function(n) {
ncname <- paste("rf", n, "_1apr_30nov.nc", sep = "")
ncin <- nc_open(ncname)
#getting variables
lat <- ncvar_get(ncin, "lat")
lon <- ncvar_get(ncin, "lon")
t <- ncvar_get(ncin, "time")

nlon <- dim(lon)
nlat <- dim(lat)
nt <- dim(t)
tunits <- ncatt_get(ncin, "time", "units")

#extracting rainfall variable details
dname <- "rf"        #lon, lat, time
rf.array <- ncvar_get(ncin, dname)
dlname <- ncatt_get(ncin, dname, "long_name")
dunits <- ncatt_get(ncin, dname, "units")
fillvalue <- ncatt_get(ncin, dname, "_FillValue")

#getting global variables
title <- ncatt_get(ncin, 0, "title")
institution <- ncatt_get(ncin, 0, "institution")
datasource <- ncatt_get(ncin, 0, "source")
references <- ncatt_get(ncin, 0, "references")
history <- ncatt_get(ncin, 0, "history")
Conventions <- ncatt_get(ncin, 0, "Conventions")

nc_close(ncin)


#splitting timeunits string into fields
tustr = strsplit(tunits$value, " ")
tdstr <- strsplit(unlist(tustr)[3], "-")
tmonth <- as.integer(unlist(tdstr)[2])
tday <- as.integer(unlist(tdstr)[3])
tyear <- as.integer(unlist(tdstr)[1])
t = t/24
#print(chron(t, origin = c(tmonth, tday, tyear)))

rf.array[rf.array == fillvalue$value] <- NA

#print(rf.array)
#print(lat)
#print(lon)

#lonsel = lon[41:50]
#print(lonsel)
#latsel = lat[23:31]
#print(latsel)
#print(rf.array[41:50, 23:31, ])

daily = means(rf.array[41:50, 23:31, ])

write.csv(daily, file = paste("dailyBNG",n, ".csv", sep = ""))
}


means <- function(s) {
	L = c()
	d = dim(s)
	for (i in 1:d[3]) {
		L = c(L, 0)
	}

	for (i in 1:d[1]) {
		for (j in 1:d[2]) {
			L = L + s[i, j, ]
		}
	}

	L = L/(d[1] * d[2])
	return(L)
}

for (i in 2008:2014) {
	open(i)
}


