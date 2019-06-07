from kl import *
print('The maximum distance between the valid Signatures is: ', validity_value)
check = input('Enter Validity Range, press Y/y to continue with validity_value: ')

if check == 'y' or check == 'Y':
	validity_value = validity_value

elif float(check) > float(validity_value):
	validity_value = float(check)

else:
	while float(check) <= validity_value:
		print('ERROR: Will fail Genuine Signatures')
		check = input('Enter Validity Range, press Y to continue with validity_value: ')

		if check != 'y' and check != 'Y' and float(check) > validity_value:
			validity_value = float(check)
			break

		elif check == 'y' or check == 'Y':
			validity_value = validity_value
			break