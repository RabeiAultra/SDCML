import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Generate universe variables
#   * Quality and service on subjective ranges [0, 10]
#   * Tip has a range of [0, 25] in units of percentage points




gray = np.arange(0, 101, 1)
variance = np.arange(0, 256, 1)
mean  = np.arange(0, 256, 1)
diagnos = np.arange(0,256,1)
#diagnos = np.arange(0,101,1)
#entropy = np.arrange(0,601,301)

# Generate fuzzy membership functions
gray_l = fuzz.trimf(gray, [0, 30, 60])
#gray_m = fuzz.trimf(gray, [25, 50, 90])
#gray_h = fuzz.trimf(gray, [65, 85, 100])

gray_m = fuzz.trimf(gray, [0, 50, 90])
gray_h = fuzz.trimf(gray, [65, 100, 100])


var_l = fuzz.trimf(variance, [ 0, 40, 85])
var_m = fuzz.trimf(variance, [ 25, 70, 120])
var_h = fuzz.trimf(variance, [65, 160,255])



mean_l = fuzz.trimf(mean, [0, 30, 65])
mean_m = fuzz.trimf(mean, [50, 65, 80])
mean_h = fuzz.trimf(mean, [65, 160, 255])

'''diagnos_healthy=fuzz.trimf(diagnos, [0, 17, 35])
diagnos_subacute=fuzz.trimf(diagnos, [23, 46, 70])
diagnos_chronic=fuzz.trimf(diagnos, [55, 82, 100])'''

diagnos_healthy=fuzz.trimf(diagnos, [0, 40, 85])
diagnos_subacute=fuzz.trimf(diagnos, [70, 120, 170])
diagnos_chronic=fuzz.trimf(diagnos, [150, 200, 255])


''''''
# Visualize these universes and membership functions
#fig, (ax0, ax1, ax2,ax3) =
fig, (ax0,ax1,ax2,ax3) =plt.subplots(nrows=4, figsize=(9, 10))

ax0.plot(gray , gray_l, 'g', linewidth=1.5, label='Low',color='green')
ax0.plot(gray , gray_m, 'b', linewidth=1.5, label='Medium',color='blue')
ax0.plot(gray , gray_h, 'r', linewidth=1.5, label='High',color = 'red')
ax0.set_title('Gray Persantage')
ax0.legend()

ax1.plot(variance, var_l, 'g', linewidth=1.5, label='Low',color='green')
ax1.plot(variance, var_m, 'b', linewidth=1.5, label='Medium',color='blue')
ax1.plot(variance, var_h, 'r', linewidth=1.5, label='High',color = 'red')
ax1.set_title('variance amount')
ax1.legend()

ax2.plot(mean, mean_l, 'g', linewidth=1.5, label='Low',color="green")
ax2.plot(mean, mean_m, 'b', linewidth=1.5, label='Medium',color='blue')
ax2.plot(mean, mean_h, 'r', linewidth=1.5, label='High',color = 'red')
ax2.set_title('mean amount')
ax2.legend()

ax3.plot(diagnos, diagnos_healthy, 'g', linewidth=1.5, label='Healthy',color="green")
ax3.plot(diagnos, diagnos_subacute, 'b', linewidth=1.5, label='SubAcute',color='blue')
ax3.plot(diagnos, diagnos_chronic, 'r', linewidth=1.5, label='Chronic',color = 'red')
ax3.set_title('Infected Amount')
ax3.legend()



    # Turn off top/right axes
for ax in (ax0,ax1,ax2,ax3):
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()
