
#include <QCoreApplication>
#include <aubio/include/aubio.h>
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    // set window size, and sampling rate
    uint_t winsize = 1024, sr = 44100;
    // create a vector
    fvec_t *this_buffer = new_fvec (winsize);
    // create the a-weighting filter
    aubio_filter_t *this_filter = new_aubio_filter_a_weighting (sr);
    while (true) {
      // here some code to put some data in this_buffer
      // ...
      // apply the filter, in place
      aubio_filter_do (this_filter, this_buffer);
      // here some code to get some data from this_buffer
      // ...
    }
    // and free the structures
    del_aubio_filter (this_filter);
    del_fvec (this_buffer);
    return a.exec();
}
