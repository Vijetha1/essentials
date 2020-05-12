import numpy as np
import random as rn
from scipy import io
import h5py
from scipy.misc import imresize as resize
import pdb
import time 

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

def get_progress_bar(i, n, s):
    if n <= 10:
        return
    if i == 10:
        print("PROGRESS:")
    if i >= 10:
        t = time.time()
        filled_progbar = round(t-s)
        full_progbar = round(n*(t-s)/i)
        if full_progbar > 100:
            quo = full_progbar//100
            full_progbar = full_progbar//quo
            filled_progbar = filled_progbar//quo
        elif full_progbar > 0:
            quo = 100/full_progbar
            full_progbar = round(full_progbar*quo)
            filled_progbar = round(filled_progbar*quo)
        print('\r'+'#'*filled_progbar + '-'*(full_progbar-filled_progbar), end="")
    elif i == 0:
        print("still computing the ETC ...")
    if i == n-1:
        print(" ")