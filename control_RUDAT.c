#include </home/lazarus/blastfirmware/libhid/hid.h>
#include <stdio.h>
#include <string.h>
#include <usb.h>
#define VENDOR_ID  0x20ce   // MiniCircuits Vendor ID
#define PRODUCT_ID 0x0023   // MiniCircuits HID RUDAT Product ID
#define PATHLEN  2
#define SEND_PACKET_LEN 64

HIDInterface* hid1;
HIDInterface* hid2;

hid_return ret;
struct usb_device *usb_dev;
struct usb_dev_handle *usb_handle;
char buffer[80], kdname[80];
const int PATH_IN[PATHLEN] = { 0x00010005, 0x00010033 };
char PACKET[SEND_PACKET_LEN];  

bool match_serial_number(struct usb_dev_handle* usbdev, void* custom, unsigned int len)
{
	bool ret;
	char* buffer = (char*)malloc(len);
	usb_get_string_simple(usbdev, usb_device(usbdev)->descriptor.iSerialNumber,buffer, len);
	ret = strncmp(buffer, (char*)custom, len) == 0;
	free(buffer);
	return ret;
}


int device_init(void)
{
	struct usb_bus *usb_bus;
	struct usb_device *dev;
	usb_init();
	int n=0;
	usb_find_busses();
	usb_find_devices();
	ret = hid_init();
	if (ret != HID_RET_SUCCESS) 
	{
	fprintf(stderr, "hid_init failed with return code %d \n", ret);
	return 1;
	}
for (usb_bus = usb_busses; usb_bus; usb_bus = usb_bus->next)
{
	for (dev = usb_bus->devices; dev; dev = dev->next)
	{
		if ((dev->descriptor.idVendor == VENDOR_ID) 
		&& (dev->descriptor.idProduct == PRODUCT_ID))
		{
///////////////////////////////////////////////

			n++;             
			usb_handle = usb_open(dev);
			int drstatus = usb_get_driver_np(usb_handle, 0, kdname,sizeof(kdname));
			if (kdname != NULL && strlen(kdname) > 0) 
				usb_detach_kernel_driver_np(usb_handle, 0);            
			usb_reset(usb_handle);
			usb_close(usb_handle);
			HIDInterfaceMatcher matcher = { VENDOR_ID, PRODUCT_ID, NULL, NULL, 0 };  
			if (n==1)
			{
				hid1 = hid_new_HIDInterface();
				if (hid1 != 0) 
				{
					ret = hid_force_open(hid1, 0, &matcher, 3);
					if (ret != HID_RET_SUCCESS) 
					{
					fprintf(stderr, "hid_force_open failed with return code %d\n", ret);                    
					}
				}
//////////////////////////////////////////////
			}
			else // n=2
			{
			hid2 = hid_new_HIDInterface();
			if (hid2 != 0) 
			{
				ret = hid_force_open(hid2, 0, &matcher, 3);
				if (ret != HID_RET_SUCCESS) 
				{
					fprintf(stderr, "hid_force_open failed with return code %d\n", ret);                    
				}
//////////////////////////////////////////////
				}
			}

		}
	
	}
}

	return 0;
}

void Get_PN (char* PNstr,HIDInterface* hid)  
{
	int i;
	char PACKETreceive[SEND_PACKET_LEN];
	PACKET[0]=40;  //     PN code
	ret = hid_interrupt_write(hid, 0x01, PACKET, SEND_PACKET_LEN,1000);
	if (ret != HID_RET_SUCCESS) {
		fprintf(stderr, "hid_interrupt_write failed with return code %d\n", ret);
	}
	ret = hid_interrupt_read(hid, 0x01, PACKETreceive, SEND_PACKET_LEN,1000);
	if (ret == HID_RET_SUCCESS) {
		strncpy(PNstr,PACKETreceive,SEND_PACKET_LEN); 
		for (i=0;PNstr[i+1]!='\0';i++) {
			PNstr[i]=PNstr[i+1];
						}						  
		PNstr[i]='\0';
	}		 
	if (ret != HID_RET_SUCCESS) {
		fprintf(stderr, "hid_interrupt_read failed with return code %d\n", ret);}
}

void Get_SN (char* SNstr,HIDInterface* hid)
{
	int i;
	char PACKETreceive[SEND_PACKET_LEN];
	PACKET[0]=41;  //     SN Code
	ret = hid_interrupt_write(hid, 0x01, PACKET, SEND_PACKET_LEN,1000);
	if (ret != HID_RET_SUCCESS) {
		fprintf(stderr, "hid_interrupt_write failed with return code %d\n", ret);
	}
	ret = hid_interrupt_read(hid, 0x01, PACKETreceive, SEND_PACKET_LEN,1000);
	if (ret == HID_RET_SUCCESS) {
		strncpy(SNstr,PACKETreceive,SEND_PACKET_LEN);
		for (i=0;SNstr[i+1]!='\0';i++) {
			SNstr[i]=SNstr[i+1];
		}  
		SNstr[i]='\0';
		}
	if (ret != HID_RET_SUCCESS) {
		fprintf(stderr, "hid_interrupt_read failed with return code %d\n", ret);	}
}

void ReadAtt (char* AttStr,HIDInterface* hid)
{
	int i;
	char PACKETreceive[SEND_PACKET_LEN];
	PACKET[0]=18;  //     Ruturn attenuation  code 
	ret = hid_interrupt_write(hid, 0x01, PACKET, SEND_PACKET_LEN,1000);
	if (ret != HID_RET_SUCCESS) {
		fprintf(stderr, "hid_interrupt_write failed with return code %d\n", ret);
	}
	ret = hid_interrupt_read(hid, 0x01, PACKETreceive, SEND_PACKET_LEN,1000);
	if (ret == HID_RET_SUCCESS) {strncpy(AttStr,PACKETreceive,SEND_PACKET_LEN);
		for(i=0;AttStr[i+1]!='\0';i++) {
			AttStr[i]=AttStr[i+1];
		}  
		AttStr[i]='\0';}
if (ret != HID_RET_SUCCESS) {
	fprintf(stderr, "hid_interrupt_read failed with return code %d\n", ret);}
}

void Set_Attenuation (float AttValue,HIDInterface* hid)
{
	int i;
	char PACKETreceive[SEND_PACKET_LEN];
	PACKET[0]=19; // Set Attenuation code is 19.
	PACKET[1]= (int)(AttValue);                  
	PACKET[2]= (int) ((AttValue-PACKET[1])*4);   
	ret = hid_interrupt_write(hid, 0x01, PACKET, SEND_PACKET_LEN,1000);
	if (ret != HID_RET_SUCCESS) {
		fprintf(stderr, "hid_interrupt_write failed with return code %d\\n", ret);
	}	
	ret = hid_interrupt_read(hid, 0x01, PACKETreceive, SEND_PACKET_LEN,1000); //  Read packet  Packetreceive[0]=1    
	if (ret != HID_RET_SUCCESS) {
		fprintf(stderr, "hid_interrupt_read failed with return code %d\n", ret);   }
}

int main(int argc, unsigned char**argv)
{
	int y=device_init();
	char AttValue1[3],AttValue2[3];
	float LastAtt=0.0;
	char PNreceive[SEND_PACKET_LEN];
	char SNreceive[SEND_PACKET_LEN];
	char RFpower[SEND_PACKET_LEN];
	int StrLen1;
	float Att1=0.0;
	float Att2=0.0;
	Get_PN(PNreceive,hid1);
	fprintf(stderr," PN1=  %s .\n",PNreceive);
	/* Get_PN(PNreceive,hid2); */
	/* fprintf(stderr," PN2=  %s .\n",PNreceive); */
	Get_SN(SNreceive,hid1);
	fprintf(stderr," SN1=  %s .\n",SNreceive);
	/* Get_SN(SNreceive,hid2); */
	/* fprintf(stderr," SN2=  %s .\n",SNreceive); */
	Att1=(float)(atof(argv[1])); 
	Set_Attenuation(Att1,hid1); // set attenuation 
	ReadAtt(AttValue1,hid1);
	LastAtt=(int)(AttValue1[0])+(float)(AttValue1[1])/4;
	fprintf(stderr," Attenuation1=  %f \n",LastAtt);
	/* Att2=(float)(atof(argv[2])); 
	Set_Attenuation(Att2,hid2);  // set attenuation 
	ReadAtt(AttValue2,hid2);
	LastAtt=(int)(AttValue2[0])+(float)(AttValue2[1])/4;
	fprintf(stderr," Attenuation2=  %f \n",LastAtt); 
	*/
	ret = hid_close(hid1);
	/* ret = hid_close(hid2); */
	if (ret != HID_RET_SUCCESS) {
		fprintf(stderr, "hid_close failed with return code %d\n", ret);
		return 1;
}
	hid_delete_HIDInterface(&hid1);
	hid_delete_HIDInterface(&hid2);
	ret = hid_cleanup();
	if (ret != HID_RET_SUCCESS) {
		fprintf(stderr, "hid_cleanup failed with return code %d\n", ret);
		return 1;
}
	return 0;
}
