import sys
import os
import csv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import navpy
from Ephemeris_Manager import EphemerisManager
import simplekml

class GNSSProcessor:
    # ---- Global Constants ----
    SECONDS_IN_WEEK = 604800
    SPEED_OF_LIGHT = 2.99792458e8
    EARTH_GRAVITY = 3.986005e14
    EARTH_ANGULAR_VELOCITY = 7.2921151467e-5
    CORRECTION_FACTOR = -4.442807633e-10

    def __init__(self, input_filepath='input_logs/driving.txt'):
        self.input_filepath = input_filepath
        if not os.path.exists(self.input_filepath):
            print(f"File {self.input_filepath} does not exist.")
            sys.exit(1)
        self.output_filepath = 'output_logs/'
        self.data_directory = 'data'
        self.ephemeris_data_directory = os.path.join(os.path.split(os.getcwd())[0], self.data_directory)
        self.android_fixes = None
        self.measurements = None

    def read_input_data(self):
        android_fixes = []
        measurements = []
        with open(self.input_filepath) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0][0] == '#':
                    if 'Fix' in row[0]:
                        android_fixes.append(row[1:])
                    elif 'Raw' in row[0]:
                        measurements.append(row[1:])
                else:
                    if row[0] == 'Fix':
                        android_fixes.append(row[1:])
                    elif row[0] == 'Raw':
                        measurements.append(row[1:])

        self.android_fixes = pd.DataFrame(android_fixes[1:], columns=android_fixes[0])
        self.measurements = pd.DataFrame(measurements[1:], columns=measurements[0])

    def preprocess_data(self):
        self.measurements['Svid'] = self.measurements['Svid'].apply(lambda x: '0' + x if len(x) == 1 else x)
        self.measurements['Constellation'] = ''
        self.measurements.loc[self.measurements['ConstellationType'] == '1', 'Constellation'] = 'G'
        self.measurements.loc[self.measurements['ConstellationType'] == '3', 'Constellation'] = 'R'
        self.measurements['satPRN'] = self.measurements['Constellation'] + self.measurements['Svid']

        self.measurements = self.measurements.loc[self.measurements['Constellation'] == 'G']

        self.measurements['Cn0DbHz'] = pd.to_numeric(self.measurements['Cn0DbHz'])
        self.measurements['TimeNanos'] = pd.to_numeric(self.measurements['TimeNanos'])
        self.measurements['FullBiasNanos'] = pd.to_numeric(self.measurements['FullBiasNanos'])
        self.measurements['ReceivedSvTimeNanos'] = pd.to_numeric(self.measurements['ReceivedSvTimeNanos'])
        self.measurements['PseudorangeRateMetersPerSecond'] = pd.to_numeric(self.measurements['PseudorangeRateMetersPerSecond'])
        self.measurements['ReceivedSvTimeUncertaintyNanos'] = pd.to_numeric(self.measurements['ReceivedSvTimeUncertaintyNanos'])

        if 'BiasNanos' in self.measurements.columns:
            self.measurements['BiasNanos'] = pd.to_numeric(self.measurements['BiasNanos'])
        else:
            self.measurements['BiasNanos'] = 0
        if 'TimeOffsetNanos' in self.measurements.columns:
            self.measurements['TimeOffsetNanos'] = pd.to_numeric(self.measurements['TimeOffsetNanos'])
        else:
            self.measurements['TimeOffsetNanos'] = 0

        self.measurements['GpsTimeNanos'] = self.measurements['TimeNanos'] - (self.measurements['FullBiasNanos'] - self.measurements['BiasNanos'])
        gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
        self.measurements['UnixTime'] = pd.to_datetime(self.measurements['GpsTimeNanos'], utc=True, origin=gpsepoch)
        self.measurements['UnixTime'] = self.measurements['UnixTime']

        self.measurements['Epoch'] = 0
        self.measurements.loc[self.measurements['UnixTime'] - self.measurements['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
        self.measurements['Epoch'] = self.measurements['Epoch'].cumsum()

    def process_gnss_data(self):
        self.read_input_data()
        self.preprocess_data()

        self.measurements['gnss_receive_time_nanoseconds'] = self.measurements['TimeNanos'] + self.measurements['TimeOffsetNanos'] - (self.measurements['FullBiasNanos'].iloc[0] + self.measurements['BiasNanos'].iloc[0])
        self.measurements['GpsWeekNumber'] = np.floor(1e-9 * self.measurements['gnss_receive_time_nanoseconds'] / GNSSProcessor.SECONDS_IN_WEEK)
        self.measurements['time_since_reference'] = 1e-9*self.measurements['gnss_receive_time_nanoseconds'] - GNSSProcessor.SECONDS_IN_WEEK * self.measurements['GpsWeekNumber']
        self.measurements['transmit_time_seconds'] = 1e-9*(self.measurements['ReceivedSvTimeNanos'] + self.measurements['TimeOffsetNanos'])
        self.measurements['pseudorange_seconds'] = self.measurements['time_since_reference'] - self.measurements['transmit_time_seconds']
        self.measurements['Pseudorange_Measurement'] = GNSSProcessor.SPEED_OF_LIGHT * self.measurements['pseudorange_seconds']

        manager = EphemerisManager(self.ephemeris_data_directory)

        epoch = 0
        num_sats = 0
        while num_sats < 5:
            one_epoch = self.measurements.loc[(self.measurements['Epoch'] == epoch) & (self.measurements['pseudorange_seconds'] < 0.1)].drop_duplicates(subset='satPRN')
            timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
            one_epoch.set_index('satPRN', inplace=True)
            num_sats = len(one_epoch.index)
            epoch += 1

        sats = one_epoch.index.unique().tolist()
        ephemeris = manager.get_ephemeris(timestamp, sats)

        def calculate_satellite_position(ephemeris, transmit_time):
            sv_position = pd.DataFrame()
            sv_position['satPRN']= ephemeris.index
            sv_position.set_index('satPRN', inplace=True)
            sv_position['GPS time'] = transmit_time - ephemeris['t_oe']
            A = ephemeris['sqrtA'].pow(2)
            n_0 = np.sqrt(GNSSProcessor.EARTH_GRAVITY / A.pow(3))
            n = n_0 + ephemeris['deltaN']
            M_k = ephemeris['M_0'] + n * sv_position['GPS time']
            E_k = M_k
            err = pd.Series(data=[1]*len(sv_position.index))
            i = 0
            while err.abs().min() > 1e-8 and i < 10:
                new_vals = M_k + ephemeris['e']*np.sin(E_k)
                err = new_vals - E_k
                E_k = new_vals
                i += 1
                
            sinE_k = np.sin(E_k)
            cosE_k = np.cos(E_k)
            delT_r = GNSSProcessor.CORRECTION_FACTOR  * ephemeris['e'].pow(ephemeris['sqrtA']) * sinE_k
            delT_oc = transmit_time - ephemeris['t_oc']
            sv_position['Sat.bias'] = ephemeris['SVclockBias'] + ephemeris['SVclockDrift'] * delT_oc + ephemeris['SVclockDriftRate'] * delT_oc.pow(2)

            v_k = np.arctan2(np.sqrt(1-ephemeris['e'].pow(2))*sinE_k,(cosE_k - ephemeris['e']))

            Phi_k = v_k + ephemeris['omega']

            sin2Phi_k = np.sin(2*Phi_k)
            cos2Phi_k = np.cos(2*Phi_k)

            du_k = ephemeris['C_us']*sin2Phi_k + ephemeris['C_uc']*cos2Phi_k
            dr_k = ephemeris['C_rs']*sin2Phi_k + ephemeris['C_rc']*cos2Phi_k
            di_k = ephemeris['C_is']*sin2Phi_k + ephemeris['C_ic']*cos2Phi_k

            u_k = Phi_k + du_k

            r_k = A*(1 - ephemeris['e']*np.cos(E_k)) + dr_k

            i_k = ephemeris['i_0'] + di_k + ephemeris['IDOT']*sv_position['GPS time']

            x_k_prime = r_k*np.cos(u_k)
            y_k_prime = r_k*np.sin(u_k)

            Omega_k = ephemeris['Omega_0'] + (ephemeris['OmegaDot'] - GNSSProcessor.EARTH_ANGULAR_VELOCITY)*sv_position['GPS time'] - GNSSProcessor.EARTH_ANGULAR_VELOCITY*ephemeris['t_oe']

            sv_position['Sat.X'] = x_k_prime*np.cos(Omega_k) - y_k_prime*np.cos(i_k)*np.sin(Omega_k)
            sv_position['Sat.Y'] = x_k_prime*np.sin(Omega_k) + y_k_prime*np.cos(i_k)*np.cos(Omega_k)
            sv_position['Sat.Z'] = y_k_prime*np.sin(i_k)
            return sv_position

        sv_position = calculate_satellite_position(ephemeris, one_epoch['transmit_time_seconds'])
        sv_position["pseudorange"] = self.measurements["Pseudorange_Measurement"] + GNSSProcessor.SPEED_OF_LIGHT * sv_position['Sat.bias']
        sv_position["cn0"] = self.measurements["Cn0DbHz"]
        sv_position = sv_position.drop(labels='Sat.bias', axis=1)
        sv_position.to_csv(os.path.join('output_logs/', 'output_xyz.csv'))

        def least_squares(xs, measured_pseudorange, x0, b0):
            dx = 100*np.ones(3)
            b = b0
            G = np.ones((measured_pseudorange.size, 4))
            iterations = 0
            while np.linalg.norm(dx) > 1e-3:
                r = np.linalg.norm(xs - x0, axis=1)
                phat = r + b0
                deltaP = measured_pseudorange - phat
                G[:, 0:3] = -(xs - x0) / r[:, None]
                sol = np.linalg.inv(np.transpose(G) @ G) @ np.transpose(G) @ deltaP
                dx = sol[0:3]
                db = sol[3]
                x0 = x0 + dx
                b0 = b0 + db
            norm_dp = np.linalg.norm(deltaP)
            return x0, b0, norm_dp

        b0 = 0
        x0 = np.array([0, 0, 0])
        xs = sv_position[['Sat.X', 'Sat.Y', 'Sat.Z']].to_numpy()
        x = x0
        b = b0
        ecef_list = []
        for epoch in self.measurements['Epoch'].unique():
            one_epoch = self.measurements.loc[(self.measurements['Epoch'] == epoch) & (self.measurements['pseudorange_seconds'] < 0.1)] 
            one_epoch = one_epoch.drop_duplicates(subset='satPRN').set_index('satPRN')
            if len(one_epoch.index) > 4:
                timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
                sats = one_epoch.index.unique().tolist()
                ephemeris = manager.get_ephemeris(timestamp, sats)
                sv_position = calculate_satellite_position(ephemeris, one_epoch['transmit_time_seconds'])

                xs = sv_position[['Sat.X', 'Sat.Y', 'Sat.Z']].to_numpy()
                pr = one_epoch['Pseudorange_Measurement'] + GNSSProcessor.SPEED_OF_LIGHT * sv_position['Sat.bias']
                pr = pr.to_numpy()

                x, b, dp = least_squares(xs, pr, x, b)
                ecef_list.append(x)

        lla = []
        lla = [navpy.ecef2lla(coord) for coord in ecef_list]

        def create_kml_file(coords, output_file):
            kml = simplekml.Kml()
            for coord in coords:
                lat, lon, alt = coord
                kml.newpoint(name="", coords=[(lon, lat, alt)])
            kml.save(output_file)

        output_file = os.path.join(self.output_filepath, 'coordinates.kml')
        create_kml_file(lla, output_file)

        with open('output_logs/lla_coordinates.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Pos.X', 'Pos.Y', 'Pos.Z', 'Lat', 'Lon', 'Alt'])
            for ecef_coord, lla_coord in zip(ecef_list, lla):
                writer.writerow([e for e in ecef_coord] + [lla_coord[0], lla_coord[1], lla_coord[2]])

def main():
    """
    There are 3 files to generate the kml file from:
    - input_logs/driving.txt
    - input_logs/fixed.txt
    - input_logs/walking.txt
    """
    valid_options = ["fixed", "walking", "driving"]
    user_input = input("Choose a file option (fixed, walking, driving): ").lower()

    # Check if the user input is valid
    if user_input not in valid_options:
        print("Please choose only from 'fixed', 'walking', or 'driving'.")
        return

    input_file_path = f"input_logs/{user_input}.txt"
    processor = GNSSProcessor(input_filepath=input_file_path)
    processor.process_gnss_data()

if __name__ == "__main__":
    main()
