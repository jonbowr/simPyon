simion.workbench_program()
adjustable max_time = 1   -- microseconds
function segment.other_actions()
  if ion_time_of_flight >= max_time then
    ion_splat = -1
  end
end