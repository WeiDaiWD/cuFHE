/**
 * Copyright 2018 Wei Dai <wdai3141@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <include/cufhe.h>
#include <include/cufhe_gpu.cuh>
#include <include/bootstrap_gpu.cuh>

namespace cufhe {

void Initialize(const PubKey& pub_key) {
  BootstrappingKeyToNTT(pub_key.bk_);
  KeySwitchingKeyToDevice(pub_key.ksk_);
}

void CleanUp() {
  DeleteBootstrappingKeyNTT();
  DeleteKeySwitchingKey();
}

inline void CtxtCopyH2D(const Ctxt& c, Stream st) {
  cudaMemcpyAsync(c.lwe_sample_device_->data(),
                  c.lwe_sample_->data(),
                  c.lwe_sample_->SizeData(),
                  cudaMemcpyHostToDevice,
                  st.st());
}

inline void CtxtCopyD2H(const Ctxt& c, Stream st) {
  cudaMemcpyAsync(c.lwe_sample_->data(),
                  c.lwe_sample_device_->data(),
                  c.lwe_sample_->SizeData(),
                  cudaMemcpyDeviceToHost,
                  st.st());
}

template <int32_t mu, int32_t space>
constexpr Torus ModSwitchToTorusConst() {
  //static const uint64_t gap = ((0x1UL << 63) / space) * 2;
  return int32_t((uint64_t(mu) * ((0x1UL << 63) / space * 2)) >> 32);
}

class NandGate {
 public:
  static __device__ inline Torus a(Torus x, Torus y) { return 0-x-y; }
  static __device__ inline Torus b(Torus x, Torus y) { return kFix+a(x, y); }
 //private:
  static const Torus kFix = ModSwitchToTorusConst<1, 8>();
};

void Nand(Ctxt& out,
          const Ctxt& in0,
          const Ctxt& in1,
          Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 8);
  if (fix != ((NandGate*)nand_gate)->kFix)
    std::cout<< "wrong\t" << fix << "\t" << ((NandGate*)nand_gate)->kFix <<std::endl;
/*  for (int i = 0; i <= in0.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = 0 - in0.lwe_sample_->data()[i]
                                   - in1.lwe_sample_->data()[i];
  out.lwe_sample_->b() += fix;
*/
  CtxtCopyH2D(in0, st);
  CtxtCopyH2D(in1, st);
  //Bootstrap(out.lwe_sample_, out.lwe_sample_, mu, st.st());
  //CtxtCopyH2D(out, st);
  //Bootstrap(out.lwe_sample_device_, out.lwe_sample_device_, mu, st.st());
  //NandBootstrap(out.lwe_sample_device_, in0.lwe_sample_device_, in1.lwe_sample_device_, mu, fix, st.st());
  Bootstrap<NandGate>(out.lwe_sample_device_, in0.lwe_sample_device_, in1.lwe_sample_device_, mu, st.st());
  CtxtCopyD2H(out, st);
}
/*
void Or(Ctxt& out,
        const Ctxt& in0,
        const Ctxt& in1,
        Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 8);
  for (int i = 0; i <= in0.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = 0 + in0.lwe_sample_->data()[i]
                                   + in1.lwe_sample_->data()[i];
  out.lwe_sample_->b() += fix;
  Bootstrap(out.lwe_sample_, out.lwe_sample_, mu, st.st());
}

void And(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  for (int i = 0; i <= in0.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = 0 + in0.lwe_sample_->data()[i]
                                   + in1.lwe_sample_->data()[i];
  out.lwe_sample_->b() += fix;
  Bootstrap(out.lwe_sample_, out.lwe_sample_, mu, st.st());
}

void Nor(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 8);
  for (int i = 0; i <= in0.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = 0 - in0.lwe_sample_->data()[i]
                                   - in1.lwe_sample_->data()[i];
  out.lwe_sample_->b() += fix;
  Bootstrap(out.lwe_sample_, out.lwe_sample_, mu, st.st());
}

void Xor(Ctxt& out,
         const Ctxt& in0,
         const Ctxt& in1,
         Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(1, 4);
  for (int i = 0; i <= in0.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = 0 + 2 * in0.lwe_sample_->data()[i]
                                   + 2 * in1.lwe_sample_->data()[i];
  out.lwe_sample_->b() += fix;
  Bootstrap(out.lwe_sample_, out.lwe_sample_, mu, st.st());
}

void Xnor(Ctxt& out,
          const Ctxt& in0,
          const Ctxt& in1,
          Stream st) {
  static const Torus mu = ModSwitchToTorus(1, 8);
  static const Torus fix = ModSwitchToTorus(-1, 4);
  for (int i = 0; i <= in0.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = 0 - 2 * in0.lwe_sample_->data()[i]
                                   - 2 * in1.lwe_sample_->data()[i];
  out.lwe_sample_->b() += fix;
  Bootstrap(out.lwe_sample_, out.lwe_sample_, mu, st.st());
}

void Not(Ctxt& out,
         const Ctxt& in,
         Stream st) {
  for (int i = 0; i <= in.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = -in.lwe_sample_->data()[i];
}

void Copy(Ctxt& out,
          const Ctxt& in,
          Stream st) {
  for (int i = 0; i <= in.lwe_sample_->n(); i ++)
    out.lwe_sample_->data()[i] = in.lwe_sample_->data()[i];
}
*/
} // namespace cufhe
