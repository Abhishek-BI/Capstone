library(gdata)
setwd("C:\\Users\\Vaibhav\\Desktop\\dir_data\\dir_data\\test")
filename = "audio_mfcc.txt"
data = read.svmlight(filename,2000)
data = as.data.frame(data)

read.svmlight <- function( filename, K ) {
  f <- file( filename, "r")
  lines <- readLines( f,skipNul = TRUE)
  close(f)
  print("File Read")
  lines = lines[!startsWith(lines, '#I')]
  
  temp = strsplit(lines,' ')
  
  y = lapply(temp, function(x) if(length(x)!=1 && length(x)!= 0 ){as.numeric(x[1])})
  print("Processing...")
  z = lapply(temp, function(x) {
        if(length(x)!=1 && length(x)!= 0 ){
        r = strsplit(x[2:length(x)],':')
        k = lapply(r, function(x) as.numeric(x[1]))
        s = lapply(r, function(x) as.numeric(x[2]) )
        mat = matrix(0,nrow=1,ncol = K)
        mat[,unlist(k)] = unlist(s)
        mat[1,]}
    } )
  print("Almost Ready...")
  data = cbind(do.call(rbind,z),unlist(y))
  
  return(data)
}


