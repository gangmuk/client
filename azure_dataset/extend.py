

rpm = [12, 51, 201, 378, 220, 64, 42, 108, 712, 178, 475, 390, 66, 231, 217, 399, 42, 128, 292, 518, 137, 239, 253, 222, 283, 70, 99, 299, 195, 100, 94, 248, 337, 38, 153, 130, 348, 99, 32, 97, 75, 137, 22, 62, 83, 243]

extended_rpm = []
for i in range(len(rpm)-1):
    extended_rpm.append(rpm[i]) 
    extended_rpm.append((rpm[i] + rpm[i+1])//2)
print(extended_rpm)