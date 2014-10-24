#ifndef Timer_H
#define Timer_H

#include <ctime>
#include <iostream>
#include <iomanip>
#include <string>

#ifndef MSGOUT
#define MSGOUT std::cout// dbg::out()
#endif

namespace Perseus
{
	namespace Utils
	{
		class Timer
		{
			friend std::ostream& operator<<(std::ostream& os, Timer& t);

		private:
			bool running;
			clock_t start_clock;
			time_t start_time;
			double acc_time;

		public:
			// 'running' is initially false.  A Timer needs to be explicitly started
			// using 'start' or 'restart'
			Timer() : running(false), start_clock(0), start_time(0), acc_time(0) { }

			void start(const char* msg = 0);
			void restart(const char* msg = 0);
			void stop(const char* msg = 0);
			void check(const char* msg = 0);
			void check(const char* msg, int msg_count);

			float elapsed_time();
		}; // class Timer

		//===========================================================================
		// Return the total time that the Timer has been in the "running"
		// state since it was first "started" or last "restarted".  For
		// "short" time periods (less than an hour), the actual cpu time
		// used is reported instead of the elapsed time.

		inline float Timer::elapsed_time()
		{
			time_t acc_sec = time(0) - start_time;
			if (acc_sec < 3600)
				return (clock() - start_clock) / (1.0f * CLOCKS_PER_SEC);
			else
				return (1.0f * acc_sec);

		} // Timer::elapsed_time

		//===========================================================================
		// Start a Timer.  If it is already running, let it continue running.
		// Print an optional message.

		inline void Timer::start(const char* msg)
		{
			// Print an optional message, something like "Starting Timer t";
			if (msg) MSGOUT  << msg << std::endl;

			// Return immediately if the Timer is already running
			if (running) return;

			// Set Timer status to running and set the start time
			running = true;
			start_clock = clock();
			start_time = time(0);

		} // Timer::start

		//===========================================================================
		// Turn the Timer off and start it again from 0.  Print an optional message.

		inline void Timer::restart(const char* msg)
		{
			// Print an optional message, something like "Restarting Timer t";
			if (msg) MSGOUT  << msg << std::endl;

			// Set Timer status to running, reset accumulated time, and set start time
			running = true;
			acc_time = 0;
			start_clock = clock();
			start_time = time(0);

		} // Timer::restart

		//===========================================================================
		// Stop the Timer and print an optional message.

		inline void Timer::stop(const char* msg)
		{
			// Print an optional message, something like "Stopping Timer t";
			if (msg) MSGOUT  << msg << std::endl;

			// Compute accumulated running time and set Timer status to not running
			if (running) acc_time += elapsed_time();
			running = false;

		} // Timer::stop

		//===========================================================================
		// Print out an optional message followed by the current Timer timing.

		inline void Timer::check(const char* msg)
		{
			std::string s;
			// Print an optional message, something like "Checking Timer t";
			if (msg) MSGOUT  << msg << " : ";

			MSGOUT  << "Time [" << std::setiosflags(std::ios::fixed)
				<< std::setprecision(4)
				<< acc_time + (running ? elapsed_time() : 0) << "] seconds\n";
		} // Timer::check
		
		inline void Timer::check(const char* msg, int msg_count)
		{
			std::string s;
			// Print an optional message, something like "Checking Timer t";
			if (msg) MSGOUT  << msg << ":";

			MSGOUT  << msg_count << ": " << "Time [" << std::setiosflags(std::ios::fixed)
				<< std::setprecision(4)
				<< acc_time + (running ? elapsed_time() : 0) << "] seconds\n";
		} // Timer::check

		//===========================================================================
		// Allow Timers to be printed to ostreams using the syntax 'os << t'
		// for an ostream 'os' and a Timer 't'.  For example, "cout << t" will
		// print out the total amount of time 't' has been "running".

		inline std::ostream& operator<<(std::ostream& os, Timer& t)
		{
			os << std::setprecision(4) << std::setiosflags(std::ios::fixed)
				<< t.acc_time + (t.running ? t.elapsed_time() : 0);
			return os;
		}
	}
}

//===========================================================================

#endif // Timer_H

