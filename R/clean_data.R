

#' Clean data
#' 
#' @param data data
#' @return cleaned dataframe
#' @export 
#' 
#' 
clean_data <- function(
  data
) {
  data |> 
    dplyr::mutate(
      DateTime = lubridate::dmy_hm(DateTime), 
      ORKTemperature = base::as.numeric(ORKTemperature), 
      ORKWindspeed= base::as.numeric(ORKWindspeed) 
    ) |> 
      tidyr::drop_na(ORKTemperature, ORKWindspeed) |> 
      dplyr::arrange(DateTime)
}
