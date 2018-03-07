import matplotlib.pyplot as plt
vecF=[0.55,49,51.63,32.71]
vecP=[84,64,52.51,22.53]
vecR=[0.2,52.51,50.80,59.64]
lr=[0.1,0.01,0.005,0.001]	

fig = plt.figure()
fig.suptitle('Performance vs learning rate', fontsize=15)
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Performance (%)', fontsize=11)
ax1.set_xlabel('Learning rate', fontsize=11)
ax1.plot(lr, vecF, c='b',label='F1-score')
ax1.plot(lr, vecP, c='r',label='Precision')
ax1.plot(lr, vecR, c='g',label='Recall')
plt.legend()
fig.savefig('perfxlr.png',bbox_inches='tight')
plt.show()
plt.close()