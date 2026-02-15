#' get smart data
#' 
#' @export
#' @return data
get_smart_data <- function(ticker, start_date = "2000-01-01", force_update = FALSE) {
  
  # Filsti: f.eks. "market_data/SPY.rds"
  file_path <- base::file.path(
    "data",
    paste0(ticker, ".rds")
  )
  
  # Tjek om filen findes, og om vi IKKE tvinger en opdatering
  if (file.exists(file_path) && !force_update) {
    message(paste("Indlæser", ticker, "fra lokal disk..."))
    return(readRDS(file_path))
  } 
  
  # Hvis filen ikke findes, hent fra Yahoo Finance
  else {
    message(paste("Downloader", ticker, "fra Yahoo Finance..."))
    
    # Hent data (auto.assign = FALSE giver os dataen direkte i variablen)
    tryCatch({
      data <- quantmod::getSymbols(ticker, src = "yahoo", from = start_date, auto.assign = FALSE)
      
      # Gem som .rds (Binært R format - lynhurtigt)
      saveRDS(data, file_path)
      
      return(data)
    }, error = function(e) {
      message(paste("Fejl ved hentning af:", ticker))
      return(NULL)
    })
  }
}