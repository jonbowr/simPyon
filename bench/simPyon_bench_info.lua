simion.workbench_program() 



function segment.initialize_run()
print(simion.wb.bounds)
simion.wb.bounds.xl =-90
print(simion.wb.bounds)

print(simion.wb.instances[1])
print(simion.wb.instances[1]:wb_to_pa_orient(1,1,1))

print(simion.wb.instances[1].pa)
print(simion.wb.instances[1].x)
simion.wb.instances[1].x = 200
end

 function segment.other_actions() 
 end