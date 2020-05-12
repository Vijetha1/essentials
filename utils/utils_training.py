import numpy as np
import random as rn
from scipy import io
import h5py
from scipy.misc import imresize as resize
import pdb

np.random.seed(42)
rn.seed(12345)

def emailSender(mystr, sendEmail=False):
	if sendEmail:
		import smtplib
		fromaddr = '****'
		toaddrs  = '****'
		SUBJECT = "From Python Program"
		message = """\
		From: %s
		To: %s
		Subject: %s

		%s
		""" % (fromaddr, ", ".join(toaddrs), SUBJECT, mystr)
		username = '****'
		password = '****'
		server = smtplib.SMTP('smtp.gmail.com:587')
		server.starttls()
		server.login(username,password)
		server.sendmail(fromaddr, toaddrs, message)
		server.quit()

def writeHashingResultsToCsv(results, fileName, mode, approaches, datasets, nBits, toCompute):
	import csv
	with open(fileName, mode) as csvfile:
		mywriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for z in range(len(toCompute)):
			mywriter.writerow([toCompute[z]])
			row = ['Approaches']
			for ii in range(len(datasets)):
				for jj in range((len(nBits))):
					if jj == 0:
						row.append(datasets[ii])
					else:
						row.append('-')
			mywriter.writerow(row)
			row = [' ']
			for i in range(len(datasets)):
				for j in range((len(nBits))):
					row.append(nBits[j])
			mywriter.writerow(row)
			for y in range(len(approaches)):
				row = []
				row.append(approaches[y])
				for x in range(len(datasets)):
					for w in range(len(nBits)):
						num = round(results[y, x, w, z], 4)
						if num != -100:
							row.append(str(num))
						else:
							row.append('-')
				mywriter.writerow(row)
			mywriter.writerow([' '])