template <typename T>
class Range {
 public:
  const T start;
  const T end;
  Range(T start, T end) : start(start), end(end) {}
  Range(T end) : start(0), end(end) {}

  std::string repr() const {
    std::ostringstream stream;
    stream << "Range<" << typeid(T).name() << ">(start=" << start << ","
           << "end=" << end << ")";
    return stream.str();
  }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Range<T> &range) {
  return os << range.repr();
}

template <typename T, size_t DIM>
class CartesianIndices {
 private:
  static std::array<T, DIM> init_multi_index(std::array<Range<T>, DIM> ranges) {
    std::array<T, DIM> multi_index;
    for (size_t k = 0; k < DIM; k++) {
      multi_index[k] = ranges[k].start;
    }
    return multi_index;
  }

 public:
  typedef std::array<T, DIM> MultiIndex;
  const std::array<Range<T>, DIM> ranges;
  CartesianIndices(std::array<Range<T>, DIM> ranges) : ranges(ranges){};

  std::string repr() const {
    std::ostringstream stream;
    stream << "CartesianIndices<" << typeid(T).name() << "," << DIM << ">(";
    for (auto range : ranges) {
      stream << range << ",";
    }
    stream << ")";
    return stream.str();
  }

  class CartesianIndex {
   private:
    std::array<T, DIM> multi_index;
    T linear_index;

   public:
    CartesianIndex() : multi_index(init_multi_index(ranges)){};

    CartesianIndex(const std::array<T, DIM> multi_index)
        : multi_index(multi_index){};

    std::string repr() const {
      std::ostringstream stream;
      stream << "CartesianIndex<" << typeid(T).name() << "," << DIM;
      for (auto i_k : multi_index) {
        stream << i_k << ",";
      }
      return stream.str();
    }

    CartesianIndex &operator++() {
      size_t k = DIM - 1;
      for (size_t kk = 0; kk < DIM; ++kk, --k) {
        multi_index[k] += 1;
        if (multi_index[k] < ranges[k].end) {
          break;
        }
        multi_index[k] = 0;
      }
      return *this;
    }
  };

  void iterate() {
    auto index = CartesianIndex();
    size_t num_iters{1};
    for (size_t k = 0; k < DIM; k++) {
      num_iters *= ranges[k].end - ranges[k].start;
    }
    for (size_t iter = 0; iter < num_iters; iter++) {
      ++index;
      std::cout << index << std::endl;
    }
  }
};

template <typename T, size_t DIM>
std::ostream &operator<<(std::ostream &os,
                         const CartesianIndices<T, DIM> &indices) {
  return os << indices.repr();
}

template <typename T, size_t DIM>
std::ostream &operator<<(
    std::ostream &os, const CartesianIndices<T, DIM>::CartesianIndex &index) {
  return os << index.repr();
}

int main(){
  std::array<Range<size_t>, 2> ranges{Range<size_t>(4), Range<size_t>(3)};
  for (auto range : ranges) {
    std::cout << range << std::endl;
  }
  CartesianIndices<size_t, 2> indices{ranges};
  std::cout << indices << std::endl;
  indices.iterate();

}