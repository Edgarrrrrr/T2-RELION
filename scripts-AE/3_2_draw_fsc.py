import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def inverse_formatter(x, pos):
    if x == 0:
        return '0'
    else:
        return '1/{:.2f}'.format(1/x)

def draw_fsc(fsc_file1, fsc_file2, output_name):
    colorset=['#6495ED','#ED6E52']

    data1=np.loadtxt(fsc_file1)
    data2=np.loadtxt(fsc_file2)
    if data1.shape[0] != data2.shape[0]:
        print("Error: The two FSC files must have the same number of rows.")
        return

    frequency1= data1[:, 0]
    fsc1= data1[:, 1]
    frequency2= data2[:, 0]
    fsc2= data2[:, 1]
    
    plt.plot(frequency1, fsc1,label="original",color=colorset[0],linewidth=3,alpha=1)
    plt.plot(frequency2, fsc2,label="optimized",color=colorset[1],linewidth=3,alpha=1)
    plt.xlabel('frequency(1/\u00C5)')
    plt.ylabel('FSC',rotation=0) 
    plt.grid(False)
    plt.ylim(0, 1)
    plt.legend(handletextpad=0.5, columnspacing=0.8,frameon=True)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(inverse_formatter))
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    plt.gca().yaxis.set_label_coords(-0, 1.04)
    plt.savefig(output_name+"_fsc.pdf", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    name_time_map={}
    if len(sys.argv) < 4:
        print("Usage: python script_AE/3_2_draw_fsc.py <fsc_file_original> <fsc_file_optimized> <output_name> ...")
        sys.exit(1)
    fsc_file1=sys.argv[1]
    fsc_file2=sys.argv[2]
    output_name=sys.argv[3]
    
    draw_fsc(fsc_file1, fsc_file2, output_name)
    